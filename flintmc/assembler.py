"""Multi-architecture assembler backed by llvm-mc.

Supports any target LLVM can assemble for — ARM Thumb-2, x86, x86_64,
AArch64, RISC-V, etc.  Provides preset profiles for common targets.

Uses the LLVM C API (via ctypes) for in-process assembly when libLLVM
is available (~0.2ms/call). Falls back to llvm-mc subprocess (~10ms/call)
otherwise.
"""

import glob
import re
import shutil
import struct
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path

from .llvm_capi import LlvmCApiBackend, try_create_backend


class AsmError(Exception):
    """Assembly failed."""


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
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, shdr_struct,
                      off_idx=4, size_idx=5)


def _extract_text_elf64(elf: bytes) -> bytes:
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x3E)[0]
    # ELF64 section header: name(4) type(4) flags(8) addr(8) offset(8) size(8) ...
    shdr_struct = struct.Struct("<IIQQQQIIQQ")
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, shdr_struct,
                      off_idx=4, size_idx=5)


def _find_text(elf: bytes, e_shoff: int, e_shentsize: int, e_shnum: int,
               e_shstrndx: int, shdr_struct: struct.Struct,
               off_idx: int, size_idx: int) -> bytes:
    """Shared logic: walk section headers, find .text, return its contents."""
    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        shdrs.append(shdr_struct.unpack_from(elf, off))

    # String table
    strtab = shdrs[e_shstrndx]
    strtab_off = strtab[off_idx]
    strtab_size = strtab[size_idx]
    strtab_data = elf[strtab_off : strtab_off + strtab_size]

    for shdr in shdrs:
        name_off = shdr[0]
        name_end = strtab_data.index(b"\x00", name_off)
        name = strtab_data[name_off:name_end].decode("ascii", errors="replace")
        if name == ".text":
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
    """
    out: list[str] = []
    for line in source.split("\n"):
        # Find the comment start (if any) — don't touch semicolons after it
        comment_start = len(line)
        in_quote = False
        for i, ch in enumerate(line):
            if ch == '"':
                in_quote = not in_quote
            elif not in_quote and ch in ("@", "#"):
                comment_start = i
                break
            elif not in_quote and i + 1 < len(line) and line[i:i+2] == "//":
                comment_start = i
                break

        # Split semicolons only in the non-comment, non-quoted prefix
        prefix = line[:comment_start]
        suffix = line[comment_start:]

        # Replace semicolons in prefix, respecting quotes
        parts: list[str] = []
        current: list[str] = []
        in_quote = False
        for ch in prefix:
            if ch == '"':
                in_quote = not in_quote
                current.append(ch)
            elif ch == ";" and not in_quote:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        parts.append("".join(current) + suffix)
        out.extend(parts)

    return "\n".join(out)


def _fix_error(stderr: str, preamble_lines: int) -> str:
    """Adjust LLVM error line numbers to account for injected preamble."""
    lines = []
    for line in stderr.splitlines():
        m = _STDIN_LINE_RE.match(line)
        if m:
            orig = int(m.group(1))
            adjusted = max(1, orig - preamble_lines)
            line = f"line {adjusted}:" + line[m.end():]
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_llvm_mc() -> str | None:
    """Try to locate llvm-mc on the system.

    Checks platform-specific install paths, then falls back to PATH.
    Returns the path string, or None if not found.

    Currently tested on macOS and Linux.  Windows support is planned
    but not yet validated — ``shutil.which`` should find llvm-mc if
    LLVM's bin directory is on PATH.
    """
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/opt/llvm/bin/llvm-mc",   # Apple Silicon Homebrew
            "/usr/local/opt/llvm/bin/llvm-mc",       # Intel Homebrew
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            "/usr/bin/llvm-mc",
            "/usr/lib/llvm/bin/llvm-mc",
        ]
        # Versioned LLVM installs: /usr/bin/llvm-mc-18, etc.
        versioned = sorted(glob.glob("/usr/bin/llvm-mc-[0-9]*"), reverse=True)
        candidates.extend(versioned)
    # win32: no well-known paths, rely on PATH below

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

    triple: str
    cpu: str
    features: str
    preamble: str
    llvm_mc: str
    backend: str

    def __init__(
        self,
        triple: str,
        *,
        cpu: str = "",
        features: str = "",
        preamble: str | None = None,
        llvm_mc: str = "",
        backend: str | None = None,
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

        # C API backend (65x faster) — uses LLVMContextSetDiagnosticHandler
        # to intercept errors instead of letting LLVM call exit().
        self._capi: LlvmCApiBackend | None = None
        if backend != "subprocess":
            self._capi = try_create_backend(triple, cpu, features)
            if backend == "capi" and self._capi is None:
                raise RuntimeError(
                    "C API backend requested but libLLVM not available"
                )

        self.backend = "capi" if self._capi is not None else "subprocess"

        if not llvm_mc:
            found = find_llvm_mc()
            if found is None and self._capi is None:
                raise FileNotFoundError(
                    "llvm-mc not found. Install LLVM or pass llvm_mc= path.\n"
                    "  macOS:  brew install llvm\n"
                    "  Linux:  apt install llvm"
                )
            self.llvm_mc = found or ""
        elif not Path(llvm_mc).is_file():
            raise FileNotFoundError(f"llvm-mc not found at: {llvm_mc}")
        else:
            self.llvm_mc = llvm_mc

        # Pre-build the subprocess command
        cmd = []
        if self.llvm_mc:
            cmd = [
                self.llvm_mc,
                f"-triple={self.triple}",
                "-filetype=obj",
                "-o", "-",
            ]
            if self.cpu:
                cmd.append(f"-mcpu={self.cpu}")
            if self.features:
                cmd.append(f"-mattr={self.features}")
        self._cmd = cmd

    def __repr__(self) -> str:
        parts = [self.triple]
        if self.cpu:
            parts.append(self.cpu)
        return f"Assembler({', '.join(parts)}, backend={self.backend})"

    def __call__(self, source: str) -> bytes:
        """Shorthand for ``asm(source)``."""
        return self.asm(source)

    # -- ARM Cortex-M profiles ---------------------------------------------

    @classmethod
    def cortex_m7_sp(cls, **kw: str) -> "Assembler":
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
    def cortex_m7_dp(cls, **kw: str) -> "Assembler":
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
    def cortex_m4(cls, **kw: str) -> "Assembler":
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
    def cortex_m33(cls, **kw: str) -> "Assembler":
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
    def cortex_m0(cls, **kw: str) -> "Assembler":
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
    def x86_64(cls, **kw: str) -> "Assembler":
        """x86-64 with Intel syntax."""
        return cls("x86_64", cpu="", features="", **kw)

    @classmethod
    def i686(cls, **kw: str) -> "Assembler":
        """x86 32-bit with Intel syntax."""
        return cls("i686", cpu="", features="", **kw)

    # -- AArch64 profiles --------------------------------------------------

    @classmethod
    def aarch64(cls, **kw: str) -> "Assembler":
        """AArch64 (ARMv8-A 64-bit)."""
        return cls("aarch64", cpu="", features="", **kw)

    # -- Core assembly -----------------------------------------------------

    def _run(self, source: str) -> bytes:
        """Assemble source text, return raw .text bytes."""
        if self._capi is not None:
            return self._run_capi(source)
        return self._run_subprocess(source)

    def _run_capi(self, source: str) -> bytes:
        """Assemble via in-process LLVM C API (~0.2ms/call).

        Uses a private LLVMContext with a diagnostic handler to intercept
        errors safely — no exit(), proper AsmError on bad input.
        """
        try:
            elf = self._capi.asm(source)  # type: ignore[union-attr]
            return _extract_text(elf)
        except RuntimeError as exc:
            raise AsmError(_fix_error(str(exc), self._preamble_lines))

    def _run_subprocess(self, source: str) -> bytes:
        """Assemble via llvm-mc subprocess (~10ms/call)."""
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
        re-invoking llvm-mc.

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
