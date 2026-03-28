import re
import struct

class AsmError(Exception):
    """Assembly failed."""

class UnsupportedArchitectureError(AsmError):
    """Raised when the requested architecture is not supported."""
    pass

_ELF_MAGIC = b"\x7fELF"

def _elf_endian(elf: bytes) -> str:
    """Return struct format prefix based on ELF EI_DATA byte."""
    ei_data = elf[5]
    if ei_data == 1: return "<"
    if ei_data == 2: return ">"
    raise AsmError(f"unsupported ELF endianness (EI_DATA={ei_data})")

def _extract_text(elf: bytes) -> bytes:
    if len(elf) < 6 or elf[:4] != _ELF_MAGIC:
        raise AsmError("llvm-mc did not produce valid ELF output")
    endian = _elf_endian(elf)
    ei_class = elf[4]
    if ei_class == 1: return _extract_text_elf32(elf, endian)
    if ei_class == 2: return _extract_text_elf64(elf, endian)
    raise AsmError(f"unsupported ELF class: {ei_class}")

def _extract_text_elf32(elf: bytes, e: str) -> bytes:
    e_shoff = struct.unpack_from(f"{e}I", elf, 0x20)[0]
    e_shentsize = struct.unpack_from(f"{e}H", elf, 0x2E)[0]
    e_shnum = struct.unpack_from(f"{e}H", elf, 0x30)[0]
    e_shstrndx = struct.unpack_from(f"{e}H", elf, 0x32)[0]
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, struct.Struct(f"{e}IIIIIIIIII"), 4, 5, e)

def _extract_text_elf64(elf: bytes, e: str) -> bytes:
    e_shoff = struct.unpack_from(f"{e}Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from(f"{e}H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from(f"{e}H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from(f"{e}H", elf, 0x3E)[0]
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, struct.Struct(f"{e}IIQQQQIIQQ"), 4, 5, e)

def _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, shdr_struct, off_idx, size_idx, e):
    strtab_shdr = shdr_struct.unpack_from(elf, e_shoff + e_shstrndx * e_shentsize)
    strtab_data = elf[strtab_shdr[off_idx] : strtab_shdr[off_idx] + strtab_shdr[size_idx]]
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name_off = struct.unpack_from(f"{e}I", elf, off)[0]
        if strtab_data[name_off : name_off + 6] == b".text\x00":
            shdr = shdr_struct.unpack_from(elf, off)
            return elf[shdr[off_idx] : shdr[off_idx] + shdr[size_idx]]
    raise AsmError("no .text section in llvm-mc output")

_STDIN_LINE_RE = re.compile(r"<(?:stdin|inline asm)>:(\d+):")

def _fix_error(stderr: str, preamble_lines: int) -> str:
    lines = []
    for line in stderr.splitlines():
        m = _STDIN_LINE_RE.match(line)
        if m:
            adjusted = max(1, int(m.group(1)) - preamble_lines)
            line = f"line {adjusted}:" + line[m.end() :]
        lines.append(line)
    return "\n".join(lines)

def _split_semicolons(source: str) -> str:
    if ";" not in source: return source
    out, in_block_comment = [], False
    for line in source.split("\n"):
        if ";" not in line and "/*" not in line and "*/" not in line:
            out.append(line)
            continue
        start, in_quote, i, n = 0, False, 0, len(line)
        while i < n:
            ch = line[i]
            if in_block_comment:
                if ch == "*" and i + 1 < n and line[i + 1] == "/": in_block_comment, i = False, i + 1
            elif ch in ('"', "'"): in_quote = not in_quote
            elif not in_quote:
                if ch == "/" and i + 1 < n and line[i + 1] == "*": in_block_comment, i = True, i + 1
                elif ch == ";": out.append(line[start:i]); start = i + 1
                elif ch in ("@", "#") or (ch == "/" and i + 1 < n and line[i + 1] == "/"): break
            i += 1
        out.append(line[start:])
    return "\n".join(out)

def _default_preamble(triple: str) -> str:
    t = triple.lower()
    if t.startswith("thumb") or (t.startswith("armv") and "m" in t.split("-")[0]): return ".syntax unified\n.thumb"
    if t.startswith("arm") and not t.startswith("arm64"): return ".syntax unified\n.arm"
    if "x86_64" in t or "x86-64" in t: return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t: return ".intel_syntax noprefix\n.code32"
    return ""
