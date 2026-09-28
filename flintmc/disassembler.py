"""Disassembly via LLVM's C API.

Uses LLVMCreateDisasmCPUFeatures / LLVMDisasmInstruction to decode
machine code back to assembly text.  Requires the same libLLVM used
by the assembler.
"""

import ctypes
import dataclasses
import threading
import weakref
from typing import Any

from .errors import AsmError, UnsupportedArchitectureError
from .common import _preset
from . import binutils
from .llvm_capi import (
    _load_llvm,
    _init_target,
    _arch_for_triple,
    _init_lock,
    VP,
    CSTR,
    SZ,
)

# Disassembler option flags (from llvm-c/Disassembler.h)
_OPT_USE_MARKUP = 1
_OPT_PRINT_IMM_HEX = 2
_OPT_ASM_PRINTER_VARIANT = 4  # Intel syntax for x86
_OPT_SET_INSTR_COMMENTS = 8
_OPT_PRINT_LATENCY = 16

def _init_disasm(lib: ctypes.CDLL, arch: str) -> None:
    """Initialize the disassembler for an architecture.

    State lives on the CDLL object so ``set_libllvm_path()`` (a new CDLL)
    gets its own argtypes and init — stale argtypes truncate pointers.
    """
    with _init_lock:
        initialized = lib.__dict__.setdefault("_flintmc_disasm_arches", set())
        if not initialized:
            _declare_disasm_argtypes(lib)
        if arch in initialized:
            return
        name = f"LLVMInitialize{arch}Disassembler"
        fn = getattr(lib, name, None)
        if fn is None:
            raise UnsupportedArchitectureError(
                f"LLVM symbol '{name}' not found. "
                f"Is {arch} disassembler support enabled in your LLVM build?"
            )
        fn.restype = None
        fn.argtypes = []
        fn()
        initialized.add(arch)


def _declare_disasm_argtypes(lib: ctypes.CDLL) -> None:
    """Declare disassembler function signatures."""
    lib.LLVMCreateDisasmCPUFeatures.restype = VP
    lib.LLVMCreateDisasmCPUFeatures.argtypes = [
        CSTR, CSTR, CSTR, VP, ctypes.c_int, VP, VP,
    ]
    lib.LLVMDisasmInstruction.restype = SZ
    lib.LLVMDisasmInstruction.argtypes = [
        VP, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
        ctypes.c_uint64, CSTR, SZ,
    ]
    lib.LLVMSetDisasmOptions.restype = ctypes.c_int
    lib.LLVMSetDisasmOptions.argtypes = [VP, ctypes.c_uint64]
    lib.LLVMDisasmDispose.restype = None
    lib.LLVMDisasmDispose.argtypes = [VP]


@dataclasses.dataclass(frozen=True, slots=True)
class DisasmInstruction:
    """A single disassembled instruction."""
    offset: int
    size: int
    code: bytes
    text: str
    address: int = 0

    @property
    def bytes(self) -> bytes:
        """Raw instruction bytes (alias for ``code``)."""
        return self.code

    @property
    def mnemonic(self) -> str:
        """Instruction mnemonic (e.g. 'mov', 'nop')."""
        return self.text.split(None, 1)[0] if self.text else ""

    @property
    def op_str(self) -> str:
        """Operand string (e.g. 'eax, 1'), empty if none."""
        parts = self.text.split(None, 1)
        return parts[1] if len(parts) > 1 else ""


class Disassembler:
    """Multi-architecture disassembler backed by LLVM's C API.

    Usage::

        dis = Disassembler.x86_64()
        for instr in dis(b'\\x90\\xc3'):
            print(instr.text)  # "nop" / "ret"

    Parameters
    ----------
    triple : str
        LLVM target triple.
    cpu : str
        Target CPU.
    features : str
        Comma-separated feature flags.
    intel : bool
        Use Intel syntax for x86 (default True).
    """

    @staticmethod
    def available() -> bool:
        """Check if the disassembler is functional (requires libLLVM)."""
        return _load_llvm() is not None

    def __init__(
        self,
        triple: str,
        *,
        cpu: str = "",
        features: str = "",
        intel: bool = True,
    ) -> None:
        self.triple = triple
        self.cpu = cpu
        self.features = features
        # No LLVM target (e.g. V850): decode with GNU objdump instead
        self._objdump = binutils.find_tool(triple, "objdump") if _arch_for_triple(triple) is None else None
        if self._objdump:
            return

        lib = _load_llvm()
        if lib is None:
            raise RuntimeError(
                "Disassembler requires libLLVM (C API). "
                "The subprocess backend does not support disassembly."
            )

        arch = _arch_for_triple(triple)
        if arch is None:
            raise UnsupportedArchitectureError(
                f"Unsupported architecture for triple: {triple}"
            )

        # Ensure both assembler and disassembler targets are initialized
        _init_target(lib, arch)
        _init_disasm(lib, arch)

        self._lib = lib
        self._local = threading.local()
        self._intel = intel

        # Minimum instruction size for skip-on-error alignment
        t = triple.lower()
        if t.startswith(("aarch64", "arm64")):
            self._min_insn_size = 4
        elif t.startswith(("thumb", "arm", "riscv")):
            self._min_insn_size = 2
        else:
            self._min_insn_size = 1

    def _get_ctx(self) -> Any:
        """Get or create the thread-local disassembler context."""
        if not hasattr(self._local, "ctx"):
            ctx = self._lib.LLVMCreateDisasmCPUFeatures(
                self.triple.encode(),
                self.cpu.encode(),
                self.features.encode(),
                None, 0, None, None,
            )
            if not ctx:
                raise RuntimeError(
                    f"LLVMCreateDisasmCPUFeatures failed for '{self.triple}'"
                )

            opts = _OPT_PRINT_IMM_HEX
            t = self.triple.lower()
            if self._intel and ("x86" in t or "i686" in t or "i386" in t):
                opts |= _OPT_ASM_PRINTER_VARIANT
            self._lib.LLVMSetDisasmOptions(ctx, opts)
            self._local.ctx = ctx
            # Dispose when this Disassembler is collected, if close() never runs
            self._local.finalizer = weakref.finalize(self, self._lib.LLVMDisasmDispose, ctx)
        return self._local.ctx

    def disasm(
        self,
        code: bytes,
        *,
        address: int = 0,
        count: int = 0,
        strict: bool = False,
    ) -> list[DisasmInstruction]:
        """Disassemble machine code bytes.

        Parameters
        ----------
        code : bytes
            Raw machine code to disassemble.
        address : int
            Base address for PC-relative display (default 0).
        count : int
            Maximum number of instructions to decode (0 = all).
        strict : bool
            If True, raise on undecodable bytes.
            If False (default), skip undecodable bytes and continue.

        Returns
        -------
        List of DisasmInstruction.

        Raises
        ------
        AsmError
            If *strict* is True and a byte cannot be decoded.
        """
        if self._objdump:
            return self._disasm_objdump(code, address, count, strict)
        ctx = self._get_ctx()
        lib = self._lib

        arr = (ctypes.c_uint8 * len(code))(*code)
        out = ctypes.create_string_buffer(512)
        results: list[DisasmInstruction] = []

        off = 0
        while off < len(code):
            if count and len(results) >= count:
                break
            ptr = ctypes.cast(
                ctypes.addressof(arr) + off,
                ctypes.POINTER(ctypes.c_uint8),
            )
            sz = lib.LLVMDisasmInstruction(
                ctx, ptr, len(code) - off, address + off, out, 512,
            )
            if sz == 0:
                if strict:
                    raise AsmError(
                        f"Cannot disassemble at offset {off}: "
                        f"0x{code[off]:02x}"
                    )
                off += self._min_insn_size
                continue
            text = out.value.decode().strip()
            results.append(DisasmInstruction(
                offset=off,
                size=sz,
                code=code[off:off + sz],
                text=text,
                address=address + off,
            ))
            off += sz

        return results

    def _disasm_objdump(self, code: bytes, address: int, count: int, strict: bool) -> list[DisasmInstruction]:
        mach = self.cpu or self.triple.split("-")[0]
        results = []
        ext = "-mextension" in self.features.split(",")  # same flag the Assembler passes to gas
        for off, raw, text in binutils.disasm(self._objdump, mach, code, address, ext):
            # objdump shows undecodable bytes as data directives or "(bad)"
            if text.startswith((".", "(bad)")):
                if strict:
                    raise AsmError(f"Cannot disassemble at offset {off}: 0x{raw[0]:02x}")
                continue
            results.append(DisasmInstruction(offset=off, size=len(raw), code=raw, text=text, address=address + off))
            if count and len(results) >= count:
                break
        return results

    def close(self) -> None:
        """Release disassembler resources for the calling thread.

        Safe to call multiple times.  After ``close()``, the disassembler
        can still be used — a fresh context is created on the next call.
        """
        if not self._objdump and hasattr(self._local, "ctx"):
            self._local.finalizer()
            del self._local.ctx, self._local.finalizer

    def __enter__(self) -> "Disassembler":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __call__(
        self,
        code: bytes,
        *,
        address: int = 0,
        count: int = 0,
        strict: bool = False,
    ) -> list[DisasmInstruction]:
        """Shorthand for ``disasm(code)``."""
        return self.disasm(code, address=address, count=count, strict=strict)

    # Alias: nyxstone-compatible name
    disassemble = disasm

    def __repr__(self) -> str:
        parts = [self.triple]
        if self.cpu:
            parts.append(self.cpu)
        return f"Disassembler({', '.join(parts)})"

    # -- Profiles (same names and settings as Assembler) ------------------

    @classmethod
    def armv6m(cls, **kw: Any) -> "Disassembler":
        """ARMv6-M (Cortex-M0/M0+)."""
        return _preset(cls, "armv6m", kw)

    @classmethod
    def armv7m(cls, **kw: Any) -> "Disassembler":
        """ARMv7-M, no FPU."""
        return _preset(cls, "armv7m", kw)

    @classmethod
    def armv8m(cls, **kw: Any) -> "Disassembler":
        """ARMv8-M Mainline, no FPU."""
        return _preset(cls, "armv8m", kw)

    @classmethod
    def cortex_m0(cls, **kw: Any) -> "Disassembler":
        """Cortex-M0/M0+."""
        return _preset(cls, "cortex_m0", kw)

    @classmethod
    def cortex_m4(cls, **kw: Any) -> "Disassembler":
        """Cortex-M4 with FPv4-SP."""
        return _preset(cls, "cortex_m4", kw)

    @classmethod
    def cortex_m7_sp(cls, **kw: Any) -> "Disassembler":
        """Cortex-M7 with FPv5-SP-D16."""
        return _preset(cls, "cortex_m7_sp", kw)

    @classmethod
    def cortex_m7_dp(cls, **kw: Any) -> "Disassembler":
        """Cortex-M7 with FPv5-D16."""
        return _preset(cls, "cortex_m7_dp", kw)

    @classmethod
    def cortex_m33(cls, **kw: Any) -> "Disassembler":
        """Cortex-M33 with FPv5-SP + DSP."""
        return _preset(cls, "cortex_m33", kw)

    @classmethod
    def x86_64(cls, **kw: Any) -> "Disassembler":
        """x86-64 (Intel syntax by default)."""
        return _preset(cls, "x86_64", kw)

    @classmethod
    def x86_32(cls, **kw: Any) -> "Disassembler":
        """x86 32-bit (Intel syntax by default)."""
        return _preset(cls, "x86_32", kw)

    i686 = x86_32

    @classmethod
    def aarch64(cls, **kw: Any) -> "Disassembler":
        """AArch64."""
        return _preset(cls, "aarch64", kw)

    @classmethod
    def riscv64(cls, **kw: Any) -> "Disassembler":
        """RISC-V 64-bit (mafd)."""
        return _preset(cls, "riscv64", kw)

    @classmethod
    def riscv32(cls, **kw: Any) -> "Disassembler":
        """RISC-V 32-bit (maf)."""
        return _preset(cls, "riscv32", kw)

    @classmethod
    def loongarch64(cls, **kw: Any) -> "Disassembler":
        """LoongArch 64-bit."""
        return _preset(cls, "loongarch64", kw)

    @classmethod
    def avr(cls, **kw: Any) -> "Disassembler":
        """Atmel AVR (avr5)."""
        return _preset(cls, "avr", kw)

    @classmethod
    def bpf(cls, **kw: Any) -> "Disassembler":
        """BPF / eBPF."""
        return _preset(cls, "bpf", kw)

    @classmethod
    def msp430(cls, **kw: Any) -> "Disassembler":
        """TI MSP430."""
        return _preset(cls, "msp430", kw)

    @classmethod
    def v850(cls, **kw: Any) -> "Disassembler":
        """Renesas V850 (base ISA). Needs ``v850-elf-objdump``."""
        return _preset(cls, "v850", kw)

    @classmethod
    def v850es(cls, **kw: Any) -> "Disassembler":
        """Renesas V850ES (V850E1 ISA). Needs ``v850-elf-objdump``."""
        return _preset(cls, "v850es", kw)

    @classmethod
    def v850e1(cls, **kw: Any) -> "Disassembler":
        """Renesas V850E1. Needs ``v850-elf-objdump``."""
        return _preset(cls, "v850e1", kw)
