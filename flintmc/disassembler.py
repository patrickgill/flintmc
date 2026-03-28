"""Disassembly via LLVM's C API.

Uses LLVMCreateDisasmCPUFeatures / LLVMDisasmInstruction to decode
machine code back to assembly text.  Requires the same libLLVM used
by the assembler.
"""

import ctypes
import dataclasses
import threading
from typing import Any

from .common import AsmError, UnsupportedArchitectureError
from .llvm_capi import (
    _load_llvm,
    _init_target,
    _arch_for_triple,
    _init_lock,
    _initialized_arches,
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

# Tracks which architectures have had their Disassembler initialized
_initialized_disasm: set[str] = set()


def _init_disasm(lib: ctypes.CDLL, arch: str) -> None:
    """Initialize the disassembler for an architecture."""
    with _init_lock:
        if arch in _initialized_disasm:
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
        _initialized_disasm.add(arch)


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


_disasm_argtypes_declared = False


@dataclasses.dataclass(frozen=True, slots=True)
class DisasmInstruction:
    """A single disassembled instruction."""
    offset: int
    size: int
    code: bytes
    text: str


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

    def __init__(
        self,
        triple: str,
        *,
        cpu: str = "",
        features: str = "",
        intel: bool = True,
    ) -> None:
        global _disasm_argtypes_declared

        lib = _load_llvm()
        if lib is None:
            raise RuntimeError("libLLVM not found")

        if not _disasm_argtypes_declared:
            _declare_disasm_argtypes(lib)
            _disasm_argtypes_declared = True

        arch = _arch_for_triple(triple)
        if arch is None:
            raise UnsupportedArchitectureError(
                f"Unsupported architecture for triple: {triple}"
            )

        # Ensure both assembler and disassembler targets are initialized
        _init_target(lib, arch)
        _init_disasm(lib, arch)

        self._lib = lib
        self.triple = triple
        self.cpu = cpu
        self.features = features
        self._local = threading.local()
        self._intel = intel

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
            if self._intel and "x86" in self.triple.lower() or "i686" in self.triple.lower() or "i386" in self.triple.lower():
                opts |= _OPT_ASM_PRINTER_VARIANT
            self._lib.LLVMSetDisasmOptions(ctx, opts)
            self._local.ctx = ctx
        return self._local.ctx

    def disasm(self, code: bytes, *, address: int = 0) -> list[DisasmInstruction]:
        """Disassemble machine code bytes.

        Parameters
        ----------
        code : bytes
            Raw machine code to disassemble.
        address : int
            Base address for PC-relative display (default 0).

        Returns
        -------
        List of DisasmInstruction(offset, size, code, text).

        Raises
        ------
        AsmError
            If a byte cannot be decoded.
        """
        ctx = self._get_ctx()
        lib = self._lib

        arr = (ctypes.c_uint8 * len(code))(*code)
        out = ctypes.create_string_buffer(512)
        results: list[DisasmInstruction] = []

        off = 0
        while off < len(code):
            ptr = ctypes.cast(
                ctypes.addressof(arr) + off,
                ctypes.POINTER(ctypes.c_uint8),
            )
            sz = lib.LLVMDisasmInstruction(
                ctx, ptr, len(code) - off, address + off, out, 512,
            )
            if sz == 0:
                raise AsmError(
                    f"Cannot disassemble at offset {off}: "
                    f"0x{code[off]:02x}"
                )
            text = out.value.decode().strip()
            results.append(DisasmInstruction(
                offset=off,
                size=sz,
                code=code[off:off + sz],
                text=text,
            ))
            off += sz

        return results

    def __call__(self, code: bytes, *, address: int = 0) -> list[DisasmInstruction]:
        """Shorthand for ``disasm(code)``."""
        return self.disasm(code, address=address)

    def __repr__(self) -> str:
        parts = [self.triple]
        if self.cpu:
            parts.append(self.cpu)
        return f"Disassembler({', '.join(parts)})"

    # -- Profiles ----------------------------------------------------------

    @classmethod
    def x86_64(cls, **kw: Any) -> "Disassembler":
        return cls("x86_64", **kw)

    @classmethod
    def x86_32(cls, **kw: Any) -> "Disassembler":
        return cls("i686", **kw)

    @classmethod
    def aarch64(cls, **kw: Any) -> "Disassembler":
        return cls("aarch64", **kw)

    @classmethod
    def cortex_m7_dp(cls, **kw: Any) -> "Disassembler":
        return cls("thumbv7em-none-eabi", cpu="cortex-m7", features="+fp-armv8,+fp64", **kw)

    @classmethod
    def riscv64(cls, **kw: Any) -> "Disassembler":
        return cls("riscv64", cpu="generic-rv64", features="+m,+a,+f,+d", **kw)

    @classmethod
    def riscv32(cls, **kw: Any) -> "Disassembler":
        return cls("riscv32", cpu="generic-rv32", features="+m,+a,+f", **kw)
