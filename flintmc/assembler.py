"""Multi-architecture assembler backed by llvm-mc.

Supports any target LLVM can assemble for — ARM Thumb-2, x86, x86_64,
AArch64, RISC-V, etc.  Provides preset profiles for common targets.

Uses the LLVM C API (via ctypes) for in-process assembly when libLLVM
is available. Falls back to llvm-mc subprocess otherwise.
"""

import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Literal, Optional

from .llvm_capi import (
    LlvmCApiBackend,
    is_available,
    set_libllvm_path,
    try_create_backend,
)

BackendType = Literal["capi", "subprocess"]


class AsmError(Exception):
    """Assembly failed."""


class UnsupportedArchitectureError(AsmError):
    """Raised when the requested architecture is not supported by flintmc or the underlying LLVM build."""
    pass


# ---------------------------------------------------------------------------
# ELF .text section extractor (32-bit and 64-bit)
# ---------------------------------------------------------------------------

_ELF_MAGIC = b"\x7fELF"


def _extract_text(elf: bytes) -> bytes:
    """Extract .text section bytes from an ELF object (32 or 64-bit)."""
    if len(elf) < 6 or elf[:4] != _ELF_MAGIC:
        raise AsmError("llvm-mc did not produce valid ELF output")

    ei_class = elf[4]  # 1 = 32-bit, 2 = 64-bit
    if ei_class == 1:
        return _extract_text_elf32(elf)
    elif ei_class == 2:
        return _extract_text_elf64(elf)
    else:
        raise AsmError(f"unsupported ELF class: {ei_class}")


def _extract_text_elf32(elf: bytes) -> bytes:
    e_shoff = struct.unpack_from("<I", elf, 0x20)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x2E)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x30)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x32)[0]
    shdr_struct = struct.Struct("<IIIIIIIIII")  # 10 x uint32
    return _find_text(
        elf,
        e_shoff,
        e_shentsize,
        e_shnum,
        e_shstrndx,
        shdr_struct,
        off_idx=4,
        size_idx=5,
    )


def _extract_text_elf64(elf: bytes) -> bytes:
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x3E)[0]
    # ELF64 section header: name(4) type(4) flags(8) addr(8) offset(8) size(8) ...
    shdr_struct = struct.Struct("<IIQQQQIIQQ")
    return _find_text(
        elf,
        e_shoff,
        e_shentsize,
        e_shnum,
        e_shstrndx,
        shdr_struct,
        off_idx=4,
        size_idx=5,
    )


def _find_text(
    elf: bytes,
    e_shoff: int,
    e_shentsize: int,
    e_shnum: int,
    e_shstrndx: int,
    shdr_struct: struct.Struct,
    off_idx: int,
    size_idx: int,
) -> bytes:
    """Shared logic: walk section headers, find .text, return its contents.

    Performance fix: Optimized to avoid full header list creation and use lazy unpacking.
    Directly jumps to the string table and only parses individual section headers
    until .text is found.
    """
    # Performance fix: Get string table header directly via e_shstrndx
    strtab_shdr = shdr_struct.unpack_from(elf, e_shoff + e_shstrndx * e_shentsize)
    strtab_off = strtab_shdr[off_idx]
    strtab_size = strtab_shdr[size_idx]
    strtab_data = elf[strtab_off : strtab_off + strtab_size]

    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        # Performance fix: sh_name is always the first 4 bytes of any Elf32/Elf64
        # section header. We only unpack this first to check for ".text".
        name_off = struct.unpack_from("<I", elf, off)[0]

        # Performance fix: Check if this name matches ".text" (null-terminated)
        # using a direct slice comparison which is faster than index/decode.
        if strtab_data[name_off : name_off + 6] == b".text\x00":
            shdr = shdr_struct.unpack_from(elf, off)
            sec_off = shdr[off_idx]
            sec_size = shdr[size_idx]
            return elf[sec_off : sec_off + sec_size]

    raise AsmError("no .text section in llvm-mc output")


# ---------------------------------------------------------------------------
# Preamble detection
# ---------------------------------------------------------------------------


def _default_preamble(triple: str) -> str:
    """Return sensible default directives for a given LLVM triple."""
    t = triple.lower()
    if t.startswith("thumb"):
        return ".syntax unified\n.thumb"
    # armv*m triples are M-profile (Thumb only, no ARM mode)
    if t.startswith("armv") and "m" in t.split("-")[0]:
        return ".syntax unified\n.thumb"
    if t.startswith("arm"):
        return ".syntax unified\n.arm"
    if "x86_64" in t or "x86-64" in t:
        return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t:
        return ".intel_syntax noprefix\n.code32"
    # AArch64, RISC-V, MIPS, etc. — no special preamble needed
    return ""


# ---------------------------------------------------------------------------
# Error line-number adjustment
# ---------------------------------------------------------------------------

# Matches both subprocess ("<stdin>:N:") and C API ("<inline asm>:N:") errors
_STDIN_LINE_RE = re.compile(r"<(?:stdin|inline asm)>:(\d+):")


def _split_semicolons(source: str) -> str:
    """Replace bare semicolons with newlines, preserving quoted strings.

    Handles double-quoted strings (``"hello;world"``) and line comments
    starting with ``@``, ``#``, or ``//``.

    Performance fix: Optimized single-pass implementation that avoids
    multiple string allocations and redundant searches.
    """
    if ";" not in source:
        return source

    out: list[str] = []
    in_block_comment = False
    for line in source.split("\n"):
        if ";" not in line and "/*" not in line and "*/" not in line:
            out.append(line)
            continue

        start = 0
        in_quote = False
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if in_block_comment:
                if ch == "*" and i + 1 < n and line[i + 1] == "/":
                    in_block_comment = False
                    i += 1
            elif ch == '"':
                in_quote = not in_quote
            elif not in_quote:
                if ch == "/" and i + 1 < n and line[i + 1] == "*":
                    in_block_comment = True
                    i += 1
                elif ch == ";":
                    out.append(line[start:i])
                    start = i + 1
                elif ch in ("@", "#") or (
                    ch == "/" and i + 1 < n and line[i + 1] == "/"
                ):
                    # Stop at line comment start
                    break
            i += 1

        # Append the rest of the line (including comment if we broke out)
        out.append(line[start:])

    return "\n".join(out)


def _fix_error(stderr: str, preamble_lines: int) -> str:
    """Adjust LLVM error line numbers to account for injected preamble."""
    lines = []
    for line in stderr.splitlines():
        m = _STDIN_LINE_RE.match(line)
        if m:
            orig = int(m.group(1))
            adjusted = max(1, orig - preamble_lines)
            line = f"line {adjusted}:" + line[m.end() :]
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_llvm_mc() -> str | None:
    """Try to locate llvm-mc on the system.

    Order of preference:
    1. LLVM_PATH environment variable (root of an LLVM installation)
    2. Platform-specific installation paths
    3. PATH via shutil.which
    """
    # 1. Check standard LLVM_PATH environment variable
    env_path = os.environ.get("LLVM_PATH")
    if env_path:
        mc_path = (
            Path(env_path)
            / "bin"
            / ("llvm-mc.exe" if sys.platform == "win32" else "llvm-mc")
        )
        if mc_path.is_file():
            return str(mc_path)

    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/opt/llvm/bin/llvm-mc",  # Apple Silicon Homebrew
            "/usr/local/opt/llvm/bin/llvm-mc",  # Intel Homebrew
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            "/usr/bin/llvm-mc",
            "/usr/lib/llvm/bin/llvm-mc",
        ]
        # Versioned LLVM installs: /usr/bin/llvm-mc-18, etc.
        versioned = sorted(glob.glob("/usr/bin/llvm-mc-[0-9]*"), reverse=True)
        candidates.extend(versioned)
    elif sys.platform == "win32":
        candidates = [
            "C:\\Program Files\\LLVM\\bin\\llvm-mc.exe",
            "C:\\Program Files (x86)\\LLVM\\bin\\llvm-mc.exe",
        ]

    for c in candidates:
        if Path(c).is_file():
            return c
    return shutil.which("llvm-mc")


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
                raise RuntimeError(f"C API backend requested but libLLVM not found")
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
            and not shutil.which(self.llvm_mc)
        ):
            raise FileNotFoundError(f"llvm-mc not found at: {self.llvm_mc}")

        # Pre-build the subprocess command
        cmd = []
        if self.llvm_mc:
            cmd = [
                self.llvm_mc,
                f"-triple={self.triple}",
                "-filetype=obj",
                "-o",
                "-",
            ]
            if self.cpu:
                cmd.append(f"-mcpu={self.cpu}")
            if self.features:
                cmd.append(f"-mattr={self.features}")
        self._cmd = cmd

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
            elf = backend.asm(source)
            return _extract_text(elf)
        except RuntimeError as exc:
            raise AsmError(_fix_error(str(exc), self._preamble_lines))

    def _run_subprocess(self, source: str) -> bytes:
        """Assemble via llvm-mc subprocess."""
        try:
            result = subprocess.run(
                self._cmd,
                input=source.encode(),
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise AsmError("llvm-mc timed out (10s limit)")

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise AsmError(_fix_error(stderr, self._preamble_lines))

        return _extract_text(result.stdout)

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

            Semicolons inside quoted strings or after ``@``/``#`` comment
            markers are preserved — only bare semicolons between
            instructions are treated as newlines.

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

        code = self._run(full)

        with self._lock:
            self._cache[source] = code
            if len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

        return code

    @property
    def cache_size(self) -> int:
        """Number of cached assembly results."""
        with self._lock:
            return len(self._cache)

    def cache_clear(self) -> None:
        """Clear the assembly result cache."""
        with self._lock:
            self._cache.clear()
