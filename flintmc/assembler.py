"""Multi-architecture assembler backed by llvm-mc.

Supports any target LLVM can assemble for — ARM Thumb-2, x86, x86_64,
AArch64, RISC-V, etc.  Provides preset profiles for common targets.

Uses the LLVM C API (via ctypes) for in-process assembly when libLLVM
is available. Falls back to llvm-mc subprocess otherwise.
"""

import dataclasses
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional

from .errors import AsmError, UnsupportedArchitectureError
from .objfile import extract_text as _extract_text
from .common import _fix_error, _split_semicolons, _default_preamble, _elf_triple, _preset
from .llvm_capi import (
    _arch_for_triple,
    is_available,
    try_create_backend,
)
from .llvm_subprocess import (
    LlvmSubprocessBackend,
    find_llvm_mc,
)

BackendType = Literal["capi", "subprocess"]

# x86 relative branches: LLVM prints the raw displacement ("jmp -3"),
# which doesn't reassemble to the same bytes in either syntax.
_X86_REL_BRANCH_RE = re.compile(r"^(?:j\w+|call\w?|loop\w*|xbegin\w?)\s+-?(?:0x)?[0-9a-f]+$")


@dataclasses.dataclass(frozen=True, slots=True)
class InstructionInfo:
    """A single assembled instruction with its position in the output."""
    offset: int
    size: int
    code: bytes
    source: str


class Assembler:
    """Multi-architecture assembler backed by llvm-mc.

    Use a preset profile to construct — ``Assembler()`` with no arguments
    is not allowed::

        asm = Assembler.x86_64()
        asm = Assembler.cortex_m7_dp()
        asm = Assembler.aarch64()

    Or pass LLVM triple/cpu/features directly for any target::

        asm = Assembler(triple="riscv64", cpu="generic-rv64",
                        features="+m,+a,+f,+d")

    Parameters
    ----------
    triple:
        LLVM target triple (e.g. ``thumbv7em-none-eabi``, ``x86_64``).
        **Required** — no default.
    cpu:
        Target CPU (e.g. ``cortex-m7``, ``generic``).
    features:
        Comma-separated LLVM feature flags (``-mattr=`` value).
    preamble:
        Assembly directives prepended to every ``asm()`` call.
        Auto-detected from the triple if not specified — ARM gets
        ``.syntax unified`` / ``.thumb``, x86 gets
        ``.intel_syntax noprefix``, etc.  Pass ``""`` to disable.
    llvm_mc:
        Path to the ``llvm-mc`` binary.  Used for subprocess fallback
        and error diagnostics.  Auto-detected if not provided.
    backend:
        Assembly backend: ``None`` (default) auto-detects — uses the
        LLVM C API if libLLVM is available (65x faster), otherwise
        falls back to llvm-mc subprocess.  Force with ``"capi"`` or
        ``"subprocess"``.
    """

    default_backend: Optional[BackendType] = None
    default_verify: bool = False
    _resolved_default: Optional[BackendType] = None

    @staticmethod
    def capi_available() -> bool:
        """Check if the LLVM C API backend is functional on this system."""
        return is_available()

    @staticmethod
    def subprocess_available() -> bool:
        """Check if the llvm-mc binary is available on this system."""
        return find_llvm_mc() is not None

    @classmethod
    def resolve_backend(cls) -> BackendType:
        """Read the currently active backend (forced or auto-detected).

        Raises RuntimeError if no backend is available.
        """
        if cls.default_backend:
            # If forced globally (e.g. via pytest flag), verify it works
            if cls.default_backend == "capi" and not cls.capi_available():
                raise RuntimeError("C API backend requested but libLLVM not found")
            if cls.default_backend == "subprocess" and not cls.subprocess_available():
                raise RuntimeError("subprocess backend requested but llvm-mc not found")
            return cls.default_backend

        if cls._resolved_default is None:
            if cls.capi_available():
                cls._resolved_default = "capi"
            elif cls.subprocess_available():
                cls._resolved_default = "subprocess"
            else:
                raise RuntimeError(
                    "No assembler backend found: libLLVM not loadable and llvm-mc not in PATH"
                )

        return cls._resolved_default

    triple: str
    cpu: str
    features: str
    preamble: str
    llvm_mc: str
    backend: BackendType

    def __init__(
        self,
        triple: str,
        *,
        cpu: str = "",
        features: str = "",
        preamble: str | None = None,
        llvm_mc: str = "",
        backend: BackendType | None = None,
    ) -> None:
        self.triple = triple
        self._obj_triple = _elf_triple(triple)
        self.cpu = cpu
        self.features = features
        self.preamble = preamble if preamble is not None else _default_preamble(triple)

        # Count preamble lines for error line-number adjustment
        self._preamble_lines = self.preamble.count("\n") + 1 if self.preamble else 0

        # Thread-safe LRU cache: source -> bytes
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_maxsize = 4096
        self._lock = threading.Lock()

        # Performance fix: Thread-Local Storage for backends.
        # This allows multiple threads to assemble concurrently by giving
        # each thread its own private libLLVM context and target machine.
        self._local = threading.local()

        # Backend selection
        # 1. Use explicit backend if passed to constructor
        # 2. Use global default (e.g. from pytest --backend)
        # 3. Auto-detect (C API preferred)
        if backend not in (None, "capi", "subprocess"):
            raise ValueError(f"backend must be 'capi' or 'subprocess', got {backend!r}")
        if backend == "capi" and not self.capi_available():
            raise RuntimeError("C API backend requested but libLLVM not found")
        self.backend = backend or self.resolve_backend()

        # Set the llvm-mc path (used for execution or diagnostics)
        self.llvm_mc = llvm_mc or find_llvm_mc() or ""
        if self.backend == "subprocess" and not self.llvm_mc:
            raise FileNotFoundError("llvm-mc not found for subprocess backend")

        if (
            self.llvm_mc
            and not Path(self.llvm_mc).is_file()
            # Explicitly not using shutil.which here as find_llvm_mc handles it
        ):
            raise FileNotFoundError(f"llvm-mc not found at: {self.llvm_mc}")

        # Eagerly validate configuration and backend availability
        try:
            self.asm("")
        except AsmError as e:
            msg = str(e)
            if "unable to get target" in msg or "unknown target" in msg:
                raise UnsupportedArchitectureError(
                    f"Unsupported or unknown architecture for triple: {self.triple}"
                ) from None
            raise

    def _get_capi_backend(self) -> Any:
        """Get or create the thread-local C API backend instance."""
        if not hasattr(self._local, "capi"):
            # Create a private backend instance for this thread
            self._local.capi = try_create_backend(self._obj_triple, self.cpu, self.features)
            if not self._local.capi:
                raise RuntimeError(f"Failed to create C API backend for {self.triple}")
        return self._local.capi

    def close(self) -> None:
        """Release backend resources for the calling thread.

        Safe to call multiple times.  After ``close()``, the assembler
        can still be used — a fresh backend is created on the next call.
        """
        if hasattr(self._local, "capi") and self._local.capi:
            self._local.capi.close()
            del self._local.capi
        if hasattr(self._local, "_verifier"):
            self._local._verifier.close()
        self.cache_clear()

    def __enter__(self) -> "Assembler":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        parts = [self.triple]
        if self.cpu:
            parts.append(self.cpu)
        return f"Assembler({', '.join(parts)}, backend={self.backend})"

    def __call__(self, source: str, *, address: int | None = None, symbols: dict[str, int] | None = None, verify: bool | None = None) -> bytes:
        """Shorthand for ``asm(source)``."""
        return self.asm(source, address=address, symbols=symbols, verify=verify)

    # -- Architecture profiles (match nyxstone shorthand names) --------------

    @classmethod
    def armv6m(cls, **kw: Any) -> "Assembler":
        """ARMv6-M (Cortex-M0/M0+). Thumb-1 only, no FPU."""
        return _preset(cls, "armv6m", kw)

    @classmethod
    def armv7m(cls, **kw: Any) -> "Assembler":
        """ARMv7-M (Cortex-M3/M4/M7). Thumb-2, no FPU by default."""
        return _preset(cls, "armv7m", kw)

    @classmethod
    def armv8m(cls, **kw: Any) -> "Assembler":
        """ARMv8-M Mainline (Cortex-M33/M55). Thumb-2 + TrustZone."""
        return _preset(cls, "armv8m", kw)

    # -- ARM Cortex-M profiles (with specific CPU/FPU) ---------------------

    @classmethod
    def cortex_m7_sp(cls, **kw: Any) -> "Assembler":
        """Cortex-M7 with FPv5-SP-D16 (single precision only).

        Rejects .f64 instructions at assembly time.  Equivalent to
        ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-sp-d16``.
        """
        return _preset(cls, "cortex_m7_sp", kw)

    @classmethod
    def cortex_m7_dp(cls, **kw: Any) -> "Assembler":
        """Cortex-M7 with FPv5-D16 (single + double precision).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-d16``.
        """
        return _preset(cls, "cortex_m7_dp", kw)

    @classmethod
    def cortex_m4(cls, **kw: Any) -> "Assembler":
        """Cortex-M4 with FPv4-SP (single precision only).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m4 -mfpu=fpv4-sp-d16``.
        """
        return _preset(cls, "cortex_m4", kw)

    @classmethod
    def cortex_m33(cls, **kw: Any) -> "Assembler":
        """Cortex-M33: ARMv8-M Mainline + FPv5-SP + DSP + TrustZone.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m33 -mfpu=fpv5-sp-d16``.
        """
        return _preset(cls, "cortex_m33", kw)

    @classmethod
    def cortex_m0(cls, **kw: Any) -> "Assembler":
        """Cortex-M0/M0+: Thumb (v6-M), no Thumb-2, no FPU.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m0``.
        """
        return _preset(cls, "cortex_m0", kw)

    # -- x86 profiles ------------------------------------------------------

    @classmethod
    def x86_64(cls, **kw: Any) -> "Assembler":
        """x86-64 with Intel syntax."""
        return _preset(cls, "x86_64", kw)

    @classmethod
    def x86_32(cls, **kw: Any) -> "Assembler":
        """x86 32-bit with Intel syntax."""
        return _preset(cls, "x86_32", kw)

    # Alias
    i686 = x86_32

    # -- AArch64 profiles --------------------------------------------------

    @classmethod
    def aarch64(cls, **kw: Any) -> "Assembler":
        """AArch64 (ARMv8-A 64-bit)."""
        return _preset(cls, "aarch64", kw)

    @classmethod
    def loongarch64(cls, **kw: Any) -> "Assembler":
        """LoongArch 64-bit."""
        return _preset(cls, "loongarch64", kw)

    # -- RISC-V profiles ---------------------------------------------------

    @classmethod
    def riscv64(cls, **kw: Any) -> "Assembler":
        """RISC-V 64-bit with common extensions (mafd)."""
        return _preset(cls, "riscv64", kw)

    @classmethod
    def riscv32(cls, **kw: Any) -> "Assembler":
        """RISC-V 32-bit with common extensions (maf)."""
        return _preset(cls, "riscv32", kw)

    # -- Other popular architectures ---------------------------------------

    @classmethod
    def avr(cls, **kw: Any) -> "Assembler":
        """Atmel AVR (8-bit microcontroller)."""
        return _preset(cls, "avr", kw)

    @classmethod
    def bpf(cls, **kw: Any) -> "Assembler":
        """BPF (Berkeley Packet Filter) / eBPF."""
        return _preset(cls, "bpf", kw)

    @classmethod
    def msp430(cls, **kw: Any) -> "Assembler":
        """TI MSP430 (ultra-low-power 16-bit MCU)."""
        return _preset(cls, "msp430", kw)

    # -- Core assembly -----------------------------------------------------

    def _run(self, source: str, header_lines: int) -> bytes:
        """Assemble source text, return the raw object file bytes."""
        if self.backend == "capi":
            return self._run_capi(source, header_lines)
        if self.backend == "subprocess":
            return self._run_subprocess(source, header_lines)

        raise AsmError(f"Unknown backend: {self.backend}")

    def _run_capi(self, source: str, header_lines: int) -> bytes:
        """Assemble via in-process LLVM C API

        Uses a private thread-local LLVMContext with a diagnostic handler to
        intercept errors safely — no exit(), proper AsmError on bad input.
        """
        try:
            # Performance fix: Retrieve the thread-local backend instance.
            # This enables true concurrency as each thread has its own
            # LLVM context and doesn't wait on a global backend lock.
            backend = self._get_capi_backend()
            return backend.asm(source)
        except RuntimeError as exc:
            raise AsmError(_fix_error(str(exc), header_lines))

    def _run_subprocess(self, source: str, header_lines: int) -> bytes:
        if not hasattr(self._local, "subprocess"):
            self._local.subprocess = LlvmSubprocessBackend(
                self._obj_triple, self.cpu, self.features, self.llvm_mc
            )
        return self._local.subprocess.asm(source, header_lines)

    _MAX_ADDRESS = 0x10_0000  # 1 MiB — keeps .org padding reasonable
    _SYMBOL_RE = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.$]*$")

    @classmethod
    def _check_address(cls, address: int | None) -> int:
        address = address or 0
        if address < 0:
            raise ValueError("address must be non-negative")
        if address > cls._MAX_ADDRESS:
            raise ValueError(
                f"address 0x{address:x} exceeds maximum 0x{cls._MAX_ADDRESS:x}. "
                "Use .org directly for larger addresses."
            )
        return address

    @staticmethod
    def _symbol_directives(symbols: dict[str, int]) -> str:
        """Build .set directives from a symbol map, validating names."""
        lines = []
        for name, addr in symbols.items():
            if not Assembler._SYMBOL_RE.match(name):
                raise ValueError(
                    f"Invalid symbol name: {name!r}. "
                    "Must match [A-Za-z_.][A-Za-z0-9_.$]*"
                )
            if not isinstance(addr, int):
                raise TypeError(f"Symbol address must be int, got {type(addr).__name__}")
            lines.append(f".set {name}, {addr}")
        return "\n".join(lines)

    def asm(self, source: str, *, address: int | None = None, symbols: dict[str, int] | None = None, verify: bool | None = None) -> bytes:
        """Assemble one or more instructions.

        Thread-safe.  Results are cached (LRU, up to 4096 entries) —
        identical source strings return the same bytes without
        re-invoking the backend.

        Parameters
        ----------
        source:
            Assembly source.  Can be a single instruction or multiple
            separated by newlines or semicolons.  The configured preamble
            (e.g. ``.syntax unified`` for ARM, ``.intel_syntax noprefix``
            for x86) is prepended automatically.

            Semicolons inside quoted strings or after ``@``/``//`` comment
            markers are preserved — only bare semicolons between
            instructions are treated as newlines.  ``#`` followed by
            a digit or sign is treated as an immediate prefix (ARM/AArch64),
            not a comment.

            Labels, literal pools, and all standard assembler directives
            are passed straight through to LLVM's MC layer.
        address:
            Base address (PC) for the assembled code.  Position-relative
            instructions (``adr``, RIP-relative, etc.) are calculated as
            if the code were placed at this address.  Max 1 MiB.
        symbols:
            Mapping of symbol names to integer values.  Injected as
            ``.set`` directives before the source.

            For data references (``mov``, ``ldr =``, etc.) the value is
            used as an absolute address or immediate::

                asm("ldr r0, =hook", symbols={"hook": 0x2000_1000})

            For branch instructions (``bl``, ``b``, ``jal``, etc.) the
            value is treated as a **byte offset from the start of the
            assembled block** — NOT an absolute address::

                asm("bl handler", symbols={"handler": 0x80})

            Symbol names must match ``[A-Za-z_.][A-Za-z0-9_.$]*``.
        verify:
            If True, disassemble the output and re-assemble it to check
            that the bytes round-trip.  Raises ``AsmError`` on mismatch.
            Defaults to ``Assembler.default_verify`` (False).
            Requires the C API backend (``libLLVM``).  Intended for
            instructions only — data directives (``.byte``, ``.word``)
            and literal pools won't round-trip.

        Returns
        -------
        The assembled machine code as bytes.

        Raises
        ------
        AsmError
            If assembly fails.
        """
        if verify is None:
            verify = self.default_verify
        if address is not None or symbols:
            code = self._assemble(source, self._check_address(address), symbols)
        else:
            with self._lock:
                if source in self._cache:
                    self._cache.move_to_end(source)
                    cached = self._cache[source]
                    if verify and cached:
                        self._verify_round_trip(cached)
                    return cached

            code = self._assemble(source)

            with self._lock:
                self._cache[source] = code
                if len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)

        if verify and code:
            self._verify_round_trip(code)

        return code

    def _verify_round_trip(self, code: bytes) -> None:
        """Disassemble code and re-assemble to verify byte-level match."""
        from .disassembler import Disassembler

        if not hasattr(self._local, "_verifier"):
            self._local._verifier = Disassembler(
                self.triple, cpu=self.cpu, features=self.features,
                intel=".intel_syntax" in self.preamble,
            )
        dis = self._local._verifier
        try:
            instrs = dis.disasm(code, strict=True)
        except AsmError:
            raise AsmError(
                f"Verification failed: cannot disassemble output ({code.hex()})"
            ) from None
        is_x86 = _arch_for_triple(self.triple) == "X86"
        # ponytail: x86 relative branches are checked by decoding only, not re-encoding
        lines = [
            ".byte " + ", ".join(map(str, i.code))
            if is_x86 and _X86_REL_BRANCH_RE.match(i.text) else i.text
            for i in instrs
        ]
        reassembled = self._assemble("\n".join(lines))
        if reassembled != code:
            raise AsmError(
                f"Verification failed: round-trip mismatch.\n"
                f"  original:    {code.hex()}\n"
                f"  reassembled: {reassembled.hex()}"
            )

    def _assemble(self, source: str, address: int = 0, symbols: dict[str, int] | None = None) -> bytes:
        """Assemble source text without caching or validation."""
        parts = [self.preamble] if self.preamble else []
        if symbols:
            parts.append(self._symbol_directives(symbols))
        if address:
            parts.append(f".org {address}")
        # Lines injected before user source, for error line-number adjustment
        header_lines = sum(p.count("\n") + 1 for p in parts)
        parts.append(_split_semicolons(source))

        code = _extract_text(self._run("\n".join(parts) + "\n", header_lines))
        return code[address:]

    def asm_each(self, source: str, *, address: int | None = None, symbols: dict[str, int] | None = None) -> list[InstructionInfo]:
        """Assemble and return per-instruction boundaries.

        Each entry has ``.offset``, ``.size``, ``.code`` (bytes), and
        ``.source`` (the input line).  Lines that produce no bytes
        (labels, ``.equ``, etc.) are skipped.  Directives that emit
        data (``.byte``, ``.align``, ``.ltorg``) DO appear.

        Assembles once with a marker label on every line, so labels,
        forward references and branch relaxation match ``asm()``.  An
        implicit literal pool at the end of the code (ARM ``ldr r0, =X``
        without ``.ltorg``) belongs to no line and is not included.
        Section switches (``.data`` etc.) are not supported.

        ``address`` and ``symbols`` work as in ``asm()``.  Offsets stay
        relative to the start of the block, not to ``address``.

        Raises ``AsmError`` if any instruction fails to assemble.
        """
        address = self._check_address(address)
        lines = [l.strip() for l in _split_semicolons(source).splitlines()]
        lines = [l for l in lines if l]
        if not lines:
            return []

        n = len(lines)
        # Marker on the same line keeps error line numbers intact.
        text = [f".Lflintmc_{i}: {line}" for i, line in enumerate(lines)]
        text.append(f".Lflintmc_{n}:")
        if _arch_for_triple(self.triple) in ("ARM", "AArch64"):
            text.append(".ltorg")  # flush literal pool before the trailer
        # Trailer: each marker's offset as 4 little-endian bytes.  Label
        # differences are assembly-time constants in every object format.
        for i in range(n + 1):
            d = f"(.Lflintmc_{i} - .Lflintmc_0)"
            text.append(f".byte {d} & 0xff, ({d} >> 8) & 0xff, ({d} >> 16) & 0xff, ({d} >> 24) & 0xff")

        try:
            code = self._assemble("\n".join(text), address, symbols)
        except AsmError:
            # Re-raise from the unmarked source so columns/echo are clean
            self._assemble("\n".join(lines), address, symbols)
            raise
        trailer = code[-4 * (n + 1):]
        offs = [int.from_bytes(trailer[4 * i:4 * i + 4], "little") for i in range(n + 1)]
        return [
            InstructionInfo(offset=offs[i], size=offs[i + 1] - offs[i],
                            code=code[offs[i]:offs[i + 1]], source=line)
            for i, line in enumerate(lines)
            if offs[i + 1] > offs[i]
        ]

    @property
    def cache_size(self) -> int:
        """Number of cached assembly results."""
        with self._lock:
            return len(self._cache)

    def cache_clear(self) -> None:
        """Clear the assembly result cache."""
        with self._lock:
            self._cache.clear()

    # Aliases
    assemble = asm
    assemble_each = asm_each
