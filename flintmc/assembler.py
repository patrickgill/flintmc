"""Multi-architecture assembler backed by llvm-mc.

Supports any target LLVM can assemble for — ARM Thumb-2, x86, x86_64,
AArch64, RISC-V, etc.  Provides preset profiles for common targets.
"""

import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


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
    if "x86_64" in t or "x86-64" in t or t.startswith("x86_64"):
        return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t or t.startswith("i686") or t.startswith("i386"):
        return ".intel_syntax noprefix\n.code32"
    # AArch64, RISC-V, MIPS, etc. — no special preamble needed
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_llvm_mc() -> str | None:
    """Try to locate llvm-mc on the system.

    Checks common Homebrew and system paths, then falls back to PATH.
    Returns the path string, or None if not found.
    """
    candidates = [
        "/opt/homebrew/opt/llvm/bin/llvm-mc",
        "/usr/local/opt/llvm/bin/llvm-mc",
        "/usr/bin/llvm-mc",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return shutil.which("llvm-mc")


@dataclass
class Assembler:
    """Multi-architecture assembler backed by llvm-mc.

    Works for any target LLVM supports.  Pass the LLVM triple, CPU,
    and feature flags — or use a preset profile like
    ``Assembler.cortex_m7_dp()`` or ``Assembler.x86_64()``.

    Parameters
    ----------
    triple:
        LLVM target triple (e.g. ``thumbv7em-none-eabi``, ``x86_64``).
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
        Path to the ``llvm-mc`` binary.  Auto-detected if not provided.

    Example
    -------
    >>> a = Assembler()
    >>> a.asm("mrs r0, PRIMASK")
    b'\\xef\\xf3\\x10\\x80'
    >>> x = Assembler.x86_64()
    >>> x.asm("mov rax, rbx; ret")
    b'\\x48\\x89\\xd8\\xc3'
    """

    triple: str = "thumbv7em-none-eabi"
    cpu: str = "cortex-m7"
    features: str = "+fp-armv8,+fp64"
    preamble: str | None = None
    llvm_mc: str = field(default="")

    def __post_init__(self) -> None:
        if not self.llvm_mc:
            found = find_llvm_mc()
            if found is None:
                raise FileNotFoundError(
                    "llvm-mc not found. Install LLVM or pass llvm_mc= path.\n"
                    "  macOS:  brew install llvm\n"
                    "  Linux:  apt install llvm"
                )
            self.llvm_mc = found
        elif not Path(self.llvm_mc).is_file():
            raise FileNotFoundError(f"llvm-mc not found at: {self.llvm_mc}")

        if self.preamble is None:
            self.preamble = _default_preamble(self.triple)

        # Assembly cache: (source, addr) -> bytes
        self._cache: dict[tuple[str, int], bytes] = {}
        # Pre-build the command prefix (immutable after init)
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

    # -- ARM Cortex-M profiles ---------------------------------------------

    @classmethod
    def cortex_m7_sp(cls, **kw) -> "Assembler":
        """Cortex-M7 with FPv5-SP-D16 (single precision only).

        Rejects .f64 instructions at assembly time.  Equivalent to
        ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-sp-d16``.
        """
        return cls(
            triple="thumbv7em-none-eabi",
            cpu="cortex-m7",
            features="+fp-armv8d16sp,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m7_dp(cls, **kw) -> "Assembler":
        """Cortex-M7 with FPv5-D16 (single + double precision).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-d16``.
        """
        return cls(
            triple="thumbv7em-none-eabi",
            cpu="cortex-m7",
            features="+fp-armv8,+fp64",
            **kw,
        )

    @classmethod
    def cortex_m4(cls, **kw) -> "Assembler":
        """Cortex-M4 with FPv4-SP (single precision only).

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m4 -mfpu=fpv4-sp-d16``.
        """
        return cls(
            triple="thumbv7em-none-eabi",
            cpu="cortex-m4",
            features="+vfp4,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m33(cls, **kw) -> "Assembler":
        """Cortex-M33: ARMv8-M Mainline + FPv5-SP + DSP + TrustZone.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m33 -mfpu=fpv5-sp-d16``.
        """
        return cls(
            triple="thumbv8m.main-none-eabi",
            cpu="cortex-m33",
            features="+fp-armv8d16sp,+dsp,-fp64,-fpregs64",
            **kw,
        )

    @classmethod
    def cortex_m0(cls, **kw) -> "Assembler":
        """Cortex-M0/M0+: Thumb (v6-M), no Thumb-2, no FPU.

        Equivalent to ``arm-none-eabi-as -mcpu=cortex-m0``.
        """
        return cls(
            triple="thumbv6m-none-eabi",
            cpu="cortex-m0",
            features="",
            **kw,
        )

    # -- x86 profiles ------------------------------------------------------

    @classmethod
    def x86_64(cls, **kw) -> "Assembler":
        """x86-64 with Intel syntax."""
        return cls(triple="x86_64", cpu="", features="", **kw)

    @classmethod
    def i686(cls, **kw) -> "Assembler":
        """x86 32-bit with Intel syntax."""
        return cls(triple="i686", cpu="", features="", **kw)

    # -- AArch64 profiles --------------------------------------------------

    @classmethod
    def aarch64(cls, **kw) -> "Assembler":
        """AArch64 (ARMv8-A 64-bit)."""
        return cls(triple="aarch64", cpu="", features="", **kw)

    # -- Core assembly -----------------------------------------------------

    def _run(self, source: str) -> bytes:
        """Assemble source text, return raw .text bytes."""
        result = subprocess.run(
            self._cmd,
            input=source.encode(),
            capture_output=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            lines = []
            for line in stderr.splitlines():
                if line.startswith("<stdin>:"):
                    parts = line.split(": ", 2)
                    lines.append(": ".join(parts[1:]) if len(parts) > 2 else line)
                else:
                    lines.append(line)
            raise AsmError("\n".join(lines))

        return _extract_text(result.stdout)

    def asm(self, source: str, addr: int = 0) -> bytes:
        """Assemble one or more instructions.

        Results are cached — identical (source, addr) pairs return the
        same bytes without re-invoking llvm-mc.

        Parameters
        ----------
        source:
            Assembly source.  Can be a single instruction or multiple
            separated by newlines or semicolons.  The configured preamble
            (e.g. ``.syntax unified`` for ARM, ``.intel_syntax noprefix``
            for x86) is prepended automatically.

            Labels, literal pools, and all standard assembler directives
            are passed straight through to LLVM's MC layer.
        addr:
            Base address for the assembled code.  Used for PC-relative
            calculations (branches, adr, etc.).

        Returns
        -------
        The assembled machine code as bytes.

        Raises
        ------
        AsmError
            If assembly fails.
        """
        key = (source, addr)
        if (cached := self._cache.get(key)) is not None:
            return cached

        # Normalize semicolons -> newlines
        text = source.replace(";", "\n")

        parts = []
        if self.preamble:
            parts.append(self.preamble)
        if addr:
            parts.append(f".org {addr:#x}")
        parts.append(text)
        full = "\n".join(parts) + "\n"

        code = self._run(full)

        # If we used .org, the output is zero-padded up to addr — strip it
        if addr:
            code = code[addr:]

        self._cache[key] = code
        return code

    def asm_one(self, mnemonic: str, addr: int = 0) -> bytes:
        """Assemble a single instruction.

        Convenience wrapper that assembles one mnemonic and returns
        the bytes.  For architectures with fixed-width instructions
        (ARM Thumb, AArch64) this also validates the output size.
        """
        code = self.asm(mnemonic, addr=addr)
        if not code:
            raise AsmError(f"no output from: {mnemonic!r}")
        return code

    @property
    def cache_size(self) -> int:
        """Number of cached assembly results."""
        return len(self._cache)

    def cache_clear(self) -> None:
        """Clear the assembly result cache."""
        self._cache.clear()
