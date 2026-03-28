import re
import struct

class AsmError(Exception):
    """Assembly failed."""

class UnsupportedArchitectureError(AsmError):
    """Raised when the requested architecture is not supported."""
    pass

_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGIC_64 = b"\xcf\xfa\xed\xfe"  # 0xFEEDFACF little-endian
_MACHO_MAGIC_32 = b"\xce\xfa\xed\xfe"  # 0xFEEDFACE little-endian
_LC_SEGMENT_64 = 0x19
_LC_SEGMENT = 0x01
# COFF machine types that indicate a valid COFF object
_COFF_MACHINES = {0x8664, 0x014c, 0xAA64, 0x01c4, 0x5064, 0x0200}

def _elf_endian(elf: bytes) -> str:
    """Return struct format prefix based on ELF EI_DATA byte."""
    ei_data = elf[5]
    if ei_data == 1: return "<"
    if ei_data == 2: return ">"
    raise AsmError(f"unsupported ELF endianness (EI_DATA={ei_data})")

def _extract_text(obj: bytes) -> bytes:
    if len(obj) < 4:
        raise AsmError("backend did not produce valid object output")
    magic = obj[:4]
    if magic == _ELF_MAGIC:
        if len(obj) < 6:
            raise AsmError("truncated ELF output")
        endian = _elf_endian(obj)
        ei_class = obj[4]
        if ei_class == 1: return _extract_text_elf32(obj, endian)
        if ei_class == 2: return _extract_text_elf64(obj, endian)
        raise AsmError(f"unsupported ELF class: {ei_class}")
    if magic == _MACHO_MAGIC_64:
        return _extract_text_macho64(obj)
    if magic == _MACHO_MAGIC_32:
        return _extract_text_macho32(obj)
    # COFF has no single magic — check machine type in first 2 bytes
    if len(obj) >= 20:
        machine = struct.unpack_from("<H", obj, 0)[0]
        if machine in _COFF_MACHINES:
            return _extract_text_coff(obj)
    raise AsmError("backend did not produce valid ELF, Mach-O, or COFF output")

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
    raise AsmError("no .text section in ELF output")

def _extract_text_macho64(obj: bytes) -> bytes:
    """Extract __text section from a 64-bit Mach-O object."""
    # mach_header_64: magic(I) cputype(I) cpusubtype(I) filetype(I)
    #                 ncmds(I) sizeofcmds(I) flags(I) reserved(I) = 32 bytes
    ncmds = struct.unpack_from("<I", obj, 16)[0]
    off = 32  # past mach_header_64
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        if cmd == _LC_SEGMENT_64:
            # segment_command_64 is 72 bytes; nsects at offset 64
            nsects = struct.unpack_from("<I", obj, off + 64)[0]
            sect_off = off + 72
            for _ in range(nsects):
                sectname = obj[sect_off:sect_off + 16].rstrip(b"\x00")
                if sectname == b"__text":
                    # section_64: addr(Q) at +32, size(Q) at +40, offset(I) at +48
                    size = struct.unpack_from("<Q", obj, sect_off + 40)[0]
                    offset = struct.unpack_from("<I", obj, sect_off + 48)[0]
                    return obj[offset:offset + size]
                sect_off += 80  # section_64 is 80 bytes
        off += cmdsize
    raise AsmError("no __text section in Mach-O output")

def _extract_text_macho32(obj: bytes) -> bytes:
    """Extract __text section from a 32-bit Mach-O object."""
    # mach_header: 28 bytes (no reserved field)
    ncmds = struct.unpack_from("<I", obj, 16)[0]
    off = 28
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        if cmd == _LC_SEGMENT:
            # segment_command is 56 bytes; nsects at offset 48
            nsects = struct.unpack_from("<I", obj, off + 48)[0]
            sect_off = off + 56
            for _ in range(nsects):
                sectname = obj[sect_off:sect_off + 16].rstrip(b"\x00")
                if sectname == b"__text":
                    # section: addr(I) at +32, size(I) at +36, offset(I) at +40
                    size = struct.unpack_from("<I", obj, sect_off + 36)[0]
                    offset = struct.unpack_from("<I", obj, sect_off + 40)[0]
                    return obj[offset:offset + size]
                sect_off += 68  # section is 68 bytes
        off += cmdsize
    raise AsmError("no __text section in Mach-O output")

def _extract_text_coff(obj: bytes) -> bytes:
    """Extract .text section from a COFF object."""
    # COFF header: Machine(H), NumberOfSections(H), TimeDateStamp(I),
    #              PointerToSymbolTable(I), NumberOfSymbols(I),
    #              SizeOfOptionalHeader(H), Characteristics(H) = 20 bytes
    nsections = struct.unpack_from("<H", obj, 2)[0]
    opt_hdr_size = struct.unpack_from("<H", obj, 16)[0]
    sect_start = 20 + opt_hdr_size
    for i in range(nsections):
        off = sect_start + i * 40
        name = obj[off:off + 8].rstrip(b"\x00")
        if name == b".text":
            rawsize = struct.unpack_from("<I", obj, off + 16)[0]
            rawptr = struct.unpack_from("<I", obj, off + 20)[0]
            return obj[rawptr:rawptr + rawsize]
    raise AsmError("no .text section in COFF output")

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
                # @  -> ARM line comment (always)
                # // -> C-style line comment
                # #  -> x86 AT&T line comment, but only when NOT followed
                #       by a digit/sign (to avoid treating ARM/AArch64
                #       immediate prefixes like #42 as comments)
                elif ch == "@" or (ch == "/" and i + 1 < n and line[i + 1] == "/") or (ch == "#" and (i + 1 >= n or line[i + 1] not in "0123456789-+")): break
            i += 1
        out.append(line[start:])
    return "\n".join(out)

_PREAMBLE_OVERRIDES: dict[str, str] = {}

def register_default_preamble(prefix: str, preamble: str) -> None:
    """Register a default preamble for triples starting with *prefix*.

    Example::

        register_default_preamble("mycpu", ".option foo")
        Assembler(triple="mycpu-none-elf")  # preamble=".option foo"

    Overrides take precedence over the built-in rules.
    """
    _PREAMBLE_OVERRIDES[prefix.lower()] = preamble

def _default_preamble(triple: str) -> str:
    t = triple.lower()
    # User-registered overrides (longest prefix first)
    for prefix, preamble in sorted(_PREAMBLE_OVERRIDES.items(), key=lambda x: len(x[0]), reverse=True):
        if t.startswith(prefix):
            return preamble
    # Built-in rules
    # Check for Cortex-M triples: armv6m, armv7m, armv8m.main, etc.
    # Match "m" in the version part (after "armv"), not in "arm" itself.
    if t.startswith("thumb") or (t.startswith("armv") and "m" in t.split("-")[0][4:]): return ".syntax unified\n.thumb"
    if t.startswith("arm") and not t.startswith("arm64"): return ".syntax unified\n.arm"
    if "x86_64" in t or "x86-64" in t: return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t: return ".intel_syntax noprefix\n.code32"
    return ""
