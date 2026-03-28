"""Multi-architecture assembler backed by llvm-mc.

Supports any target LLVM can assemble for — ARM Thumb-2, x86, x86_64,
AArch64, RISC-V, etc.  Provides preset profiles for common targets.

Uses the LLVM C API (via ctypes) for in-process assembly when libLLVM
is available. Falls back to llvm-mc subprocess otherwise.
"""

import dataclasses
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional

from .common import (
    AsmError,
    UnsupportedArchitectureError,
    _extract_text,
    _fix_error,
    _split_semicolons,
    _default_preamble,
)
from .llvm_capi import (
    is_available,
    try_create_backend,
)
from .llvm_subprocess import (
    LlvmSubprocessBackend,
    find_llvm_mc,
)

BackendType = Literal["capi", "subprocess"]


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
                    "No assembler backend found. Install LLVM (libLLVM + llvm-mc).\n"
                    "  macOS: brew install llvm\n"
                    "  Linux: apt install llvm"
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
        requested = backend or Assembler.default_backend

        if requested == "capi":
            if not self.capi_available():
                raise RuntimeError("C API backend requested but libLLVM not found")
            self.backend = "capi"
        elif requested == "subprocess":
            self.backend = "subprocess"
        else:
            # Auto-detect: Prefer C API
            if self.capi_available():
                self.backend = "capi"
            elif self.subprocess_available():
                self.backend = "subprocess"
            else:
                raise RuntimeError(f"No assembler backend found for {triple}")

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
            self._local.capi = try_create_backend(self.triple, self.cpu, self.features)
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

    def __call__(self, source: str) -> bytes:
        """Shorthand for ``asm(source)``."""
        return self.asm(source)

    # -- Architecture profiles (match nyxstone shorthand names) --------------

    @classmethod
    def armv6m(cls, **kw: Any) -> "Assembler":
        """ARMv6-M (Cortex-M0/M0+). Thumb-1 only, no FPU."""
        return cls("armv6m-none-eabi", cpu="", features="", **kw)

    @classmethod
    def armv7m(cls, **kw: Any) -> "Assembler":
        """ARMv7-M (Cortex-M3/M4/M7). Thumb-2, no FPU by default."""
        return cls("armv7m-none-eabi", cpu="", features="", **kw)

    @classmethod
    def armv8m(cls, **kw: Any) -> "Assembler":
        """ARMv8-M Mainline (Cortex-M33/M55). Thumb-2 + TrustZone."""
        return cls("armv8m.main-none-eabi", cpu="", features="", **kw)

    # -- ARM Cortex-M profiles (with specific CPU/FPU) ---------------------

    @classmethod
    def cortex_m7_sp(cls, **kw: Any) -> "Assembler":
        """Cortex-M7 with FPv5-SP-D16 (single precision only).

        Rejects .f64 instructions at assembly time.  Equivalent to
        ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-sp-d16``.
        """
        return cls(
            "thumbv7em-none-eabi",
            cpu="cortex-m7",
            features="+fp-armv8d16sp,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m7_dp(cls, **kw: Any) -> "Assembler":
        """Cortex-M7 with FPv5-D16 (single + double precision).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-d16``.
        """
        return cls(
            "thumbv7em-none-eabi",
            cpu="cortex-m7",
            features="+fp-armv8,+fp64",
            **kw,
        )

    @classmethod
    def cortex_m4(cls, **kw: Any) -> "Assembler":
        """Cortex-M4 with FPv4-SP (single precision only).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m4 -mfpu=fpv4-sp-d16``.
        """
        return cls(
            "thumbv7em-none-eabi",
            cpu="cortex-m4",
            features="+vfp4,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m33(cls, **kw: Any) -> "Assembler":
        """Cortex-M33: ARMv8-M Mainline + FPv5-SP + DSP + TrustZone.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m33 -mfpu=fpv5-sp-d16``.
        """
        return cls(
            "thumbv8m.main-none-eabi",
            cpu="cortex-m33",
            features="+fp-armv8d16sp,+dsp,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m0(cls, **kw: Any) -> "Assembler":
        """Cortex-M0/M0+: Thumb (v6-M), no Thumb-2, no FPU.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m0``.
        """
        return cls(
            "thumbv6m-none-eabi",
            cpu="cortex-m0",
            features="",
            **kw,
        )

    # -- x86 profiles ------------------------------------------------------

    @classmethod
    def x86_64(cls, **kw: Any) -> "Assembler":
        """x86-64 with Intel syntax."""
        return cls("x86_64", cpu="", features="", **kw)

    @classmethod
    def x86_32(cls, **kw: Any) -> "Assembler":
        """x86 32-bit with Intel syntax."""
        return cls("i686", cpu="", features="", **kw)

    # Alias
    i686 = x86_32

    # -- AArch64 profiles --------------------------------------------------

    @classmethod
    def aarch64(cls, **kw: Any) -> "Assembler":
        """AArch64 (ARMv8-A 64-bit)."""
        return cls("aarch64", cpu="", features="", **kw)

    @classmethod
    def loongarch64(cls, **kw: Any) -> "Assembler":
        """LoongArch 64-bit."""
        return cls("loongarch64", **kw)

    # -- RISC-V profiles ---------------------------------------------------

    @classmethod
    def riscv64(cls, **kw: Any) -> "Assembler":
        """RISC-V 64-bit with common extensions (mafd)."""
        return cls("riscv64", cpu="generic-rv64", features="+m,+a,+f,+d", **kw)

    @classmethod
    def riscv32(cls, **kw: Any) -> "Assembler":
        """RISC-V 32-bit with common extensions (maf)."""
        return cls("riscv32", cpu="generic-rv32", features="+m,+a,+f", **kw)

    # -- Other popular architectures ---------------------------------------

    @classmethod
    def avr(cls, **kw: Any) -> "Assembler":
        """Atmel AVR (8-bit microcontroller)."""
        return cls("avr", cpu="avr5", **kw)

    @classmethod
    def bpf(cls, **kw: Any) -> "Assembler":
        """BPF (Berkeley Packet Filter) / eBPF."""
        return cls("bpf", **kw)

    @classmethod
    def msp430(cls, **kw: Any) -> "Assembler":
        """TI MSP430 (ultra-low-power 16-bit MCU)."""
        return cls("msp430", **kw)

    # -- Core assembly -----------------------------------------------------

    def _run(self, source: str) -> bytes:
        """Assemble source text, return raw .text bytes."""
        if self.backend == "capi":
            return self._run_capi(source)
        if self.backend == "subprocess":
            return self._run_subprocess(source)

        raise AsmError(f"Unknown backend: {self.backend}")

    def _run_capi(self, source: str) -> bytes:
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
            raise AsmError(_fix_error(str(exc), self._preamble_lines))

    def _run_subprocess(self, source: str) -> bytes:
        if not hasattr(self._local, "subprocess"):
            self._local.subprocess = LlvmSubprocessBackend(
                self.triple, self.cpu, self.features, self.llvm_mc
            )
        return self._local.subprocess.asm(source, self._preamble_lines)

    def asm(self, source: str) -> bytes:
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

        Returns
        -------
        The assembled machine code as bytes.

        Raises
        ------
        AsmError
            If assembly fails.
        """
        with self._lock:
            if source in self._cache:
                self._cache.move_to_end(source)
                return self._cache[source]

        # Normalize bare semicolons -> newlines (preserve quoted/commented)
        text = _split_semicolons(source)

        parts = []
        if self.preamble:
            parts.append(self.preamble)
        parts.append(text)
        full = "\n".join(parts) + "\n"

        elf = self._run(full)
        code = _extract_text(elf)

        with self._lock:
            self._cache[source] = code
            if len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

        return code

    def asm_each(self, source: str) -> list[InstructionInfo]:
        """Assemble and return per-instruction boundaries.

        Each entry has ``.offset``, ``.size``, ``.code`` (bytes), and
        ``.source`` (the input line).  Labels and directives are
        accumulated as context but don't appear in the output list.

        Uses cumulative assembly: assembles lines 1..N progressively
        to determine where each instruction's bytes fall.  This
        correctly handles labels and forward references.

        Raises ``AsmError`` if any instruction fails to assemble.
        """
        text = _split_semicolons(source)
        lines = [l.strip() for l in text.splitlines()]
        lines = [l for l in lines if l]

        preamble_parts = [self.preamble] if self.preamble else []
        results: list[InstructionInfo] = []
        prev_size = 0
        accumulated: list[str] = []

        for line in lines:
            accumulated.append(line)
            full = "\n".join(preamble_parts + accumulated) + "\n"
            elf = self._run(full)
            code = _extract_text(elf)
            cur_size = len(code)

            if cur_size > prev_size:
                instr_bytes = code[prev_size:]
                results.append(InstructionInfo(
                    offset=prev_size,
                    size=cur_size - prev_size,
                    code=instr_bytes,
                    source=line,
                ))
                prev_size = cur_size
            # else: label, directive, or alignment — no new bytes

        return results

    @property
    def cache_size(self) -> int:
        """Number of cached assembly results."""
        with self._lock:
            return len(self._cache)

    def cache_clear(self) -> None:
        """Clear the assembly result cache."""
        with self._lock:
            self._cache.clear()
