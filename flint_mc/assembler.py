"""ARM Thumb-2 assembler using llvm-mc as backend.

Replaces keystone-engine for Cortex-M assembly — handles M-class special
registers (PRIMASK, BASEPRI, FAULTMASK) in MSR/MRS, FPv5 instructions
(VFMA etc.), and everything else LLVM's ARM backend supports.
"""

import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class AsmError(Exception):
    """Assembly failed."""


# ---------------------------------------------------------------------------
# Minimal ELF .text section extractor (32-bit little-endian only)
# ---------------------------------------------------------------------------

_ELF_MAGIC = b"\x7fELF"
_ELF_HDR = struct.Struct("<4s5x7x2xI5xHHH")  # magic,_,_,e_type,...,e_shoff,...,e_shentsize,e_shnum,e_shstrndx
_ELF32_SHDR = struct.Struct("<IIIIIIIIII")  # 10 × uint32


def _extract_text(elf: bytes) -> bytes:
    """Extract .text section bytes from a minimal 32-bit ELF object."""
    if elf[:4] != _ELF_MAGIC:
        raise AsmError("llvm-mc did not produce valid ELF output")

    # ELF32 header fields we need
    e_shoff = struct.unpack_from("<I", elf, 0x20)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x2E)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x30)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x32)[0]

    # Read section headers
    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        vals = _ELF32_SHDR.unpack_from(elf, off)
        shdrs.append(vals)
        # vals: (sh_name, sh_type, sh_flags, sh_addr, sh_offset,
        #        sh_size, sh_link, sh_info, sh_addralign, sh_entsize)

    # String table for section names
    strtab_shdr = shdrs[e_shstrndx]
    strtab_off = strtab_shdr[4]
    strtab_size = strtab_shdr[5]
    strtab = elf[strtab_off : strtab_off + strtab_size]

    # Find .text
    for shdr in shdrs:
        name_off = shdr[0]
        name_end = strtab.index(b"\x00", name_off)
        name = strtab[name_off:name_end].decode("ascii", errors="replace")
        if name == ".text":
            sec_off = shdr[4]
            sec_size = shdr[5]
            return elf[sec_off : sec_off + sec_size]

    raise AsmError("no .text section in llvm-mc output")


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
    """ARM Thumb-2 assembler backed by llvm-mc.

    Parameters
    ----------
    triple:
        LLVM target triple.  Default targets Cortex-M with hard-float.
    cpu:
        Target CPU.  Default ``cortex-m7`` gives full Thumb-2 + FPv5.
    features:
        Comma-separated LLVM feature flags.
    llvm_mc:
        Path to the ``llvm-mc`` binary.  Auto-detected if not provided.

    Example
    -------
    >>> a = Assembler()
    >>> a.asm("mrs r0, PRIMASK")
    b'\\xef\\xf3\\x10\\x80'
    """

    triple: str = "thumbv7em-none-eabi"
    cpu: str = "cortex-m7"
    features: str = "+fp-armv8"
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

    def _run(self, source: str) -> bytes:
        """Assemble source text, return raw .text bytes."""
        cmd = [
            self.llvm_mc,
            f"-triple={self.triple}",
            f"-mcpu={self.cpu}",
            f"-mattr={self.features}",
            "-filetype=obj",
            "-o", "-",
        ]

        result = subprocess.run(
            cmd,
            input=source.encode(),
            capture_output=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            # Strip the "<stdin>:N:N: " prefix for cleaner errors
            lines = []
            for line in stderr.splitlines():
                if line.startswith("<stdin>:"):
                    # "<stdin>:3:1: error: ..." → "error: ..."
                    parts = line.split(": ", 2)
                    lines.append(": ".join(parts[1:]) if len(parts) > 2 else line)
                else:
                    lines.append(line)
            raise AsmError("\n".join(lines))

        return _extract_text(result.stdout)

    def asm(self, source: str, addr: int = 0) -> bytes:
        """Assemble one or more ARM Thumb instructions.

        Parameters
        ----------
        source:
            Assembly source.  Can be a single instruction or multiple
            separated by newlines or semicolons.  The ``.syntax unified``
            and ``.thumb`` directives are prepended automatically.
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
        # Normalize semicolons → newlines
        text = source.replace(";", "\n")

        lines = [".syntax unified", ".thumb"]
        if addr:
            lines.append(f".org {addr:#x}")
        lines.append(text)
        full = "\n".join(lines) + "\n"

        code = self._run(full)

        # If we used .org, the output is zero-padded up to addr — strip it
        if addr:
            code = code[addr:]

        return code

    def asm_one(self, mnemonic: str, addr: int = 0) -> bytes:
        """Assemble a single instruction.

        Convenience wrapper — identical to ``asm()`` but asserts the
        result is a single 2-byte or 4-byte Thumb instruction.
        """
        code = self.asm(mnemonic, addr=addr)
        if len(code) not in (2, 4):
            raise AsmError(
                f"expected single instruction (2 or 4 bytes), "
                f"got {len(code)} bytes from: {mnemonic!r}"
            )
        return code
