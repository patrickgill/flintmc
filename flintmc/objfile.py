"""Extract .text / __text section from ELF, Mach-O, and COFF objects.

Pure-Python parsers for the three object file formats LLVM can emit.
Reads section header tables to locate the code section and optionally
applies relocations for resolved symbol references.
"""

import struct

from .errors import AsmError

_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGIC_64 = b"\xcf\xfa\xed\xfe"  # 0xFEEDFACF little-endian
_MACHO_MAGIC_32 = b"\xce\xfa\xed\xfe"  # 0xFEEDFACE little-endian
_LC_SEGMENT_64 = 0x19
_LC_SEGMENT = 0x01
# COFF machine types that indicate a valid COFF object
_COFF_MACHINES = {0x8664, 0x014c, 0xAA64, 0x01c4, 0x5064, 0x0200}

_SHT_RELA, _SHT_REL = 4, 9
_SHN_UNDEF, _SHN_ABS = 0, 0xFFF1

# How to resolve each (e_machine, r_type) left in .text.  Anything not listed
# raises: shipping the unpatched field would be silently wrong code.
#   pc32   32-bit S + A - P          abs32 / abs64   S + A
#   a64    AArch64 immediate field (see _A64)       ignore   linker hint, bytes final
_EM_386, _EM_ARM, _EM_X86_64, _EM_AARCH64, _EM_RISCV = 3, 40, 62, 183, 243
_RELOCS = {
    (_EM_X86_64, 0): "ignore", (_EM_X86_64, 1): "abs64", (_EM_X86_64, 2): "pc32",
    (_EM_X86_64, 4): "pc32", (_EM_X86_64, 10): "abs32", (_EM_X86_64, 11): "abs32",
    (_EM_386, 0): "ignore", (_EM_386, 1): "abs32", (_EM_386, 2): "pc32",
    (_EM_ARM, 0): "ignore", (_EM_ARM, 2): "abs32", (_EM_ARM, 40): "ignore",
    (_EM_AARCH64, 0): "ignore", (_EM_AARCH64, 257): "abs64", (_EM_AARCH64, 258): "abs32",
    (_EM_RISCV, 0): "ignore", (_EM_RISCV, 1): "abs32", (_EM_RISCV, 2): "abs64",
    (_EM_RISCV, 43): "ignore", (_EM_RISCV, 51): "ignore",
}

# AArch64 instruction relocations: r_type -> (value, right shift, field width, field lsb)
#   pc: S+A-P   page: Page(S+A) - Page(P)   lo12: (S+A) & 0xfff (no range check)
# adr-format fields (adr/adrp) split imm21 into immlo[30:29] and immhi[23:5].
_A64 = {
    273: ("pc", 2, 19, 5),     # LD_PREL_LO19      ldr literal
    274: ("pc", 0, 21, None),  # ADR_PREL_LO21     adr
    275: ("page", 12, 21, None),  # ADR_PREL_PG_HI21  adrp
    277: ("lo12", 0, 12, 10),  # ADD_ABS_LO12_NC
    278: ("lo12", 0, 12, 10),  # LDST8_ABS_LO12_NC
    279: ("pc", 2, 14, 5),     # TSTBR14           tbz/tbnz
    280: ("pc", 2, 19, 5),     # CONDBR19          b.cond/cbz/cbnz
    282: ("pc", 2, 26, 0),     # JUMP26            b
    283: ("pc", 2, 26, 0),     # CALL26            bl
    284: ("lo12", 1, 12, 10),  # LDST16_ABS_LO12_NC
    285: ("lo12", 2, 12, 10),  # LDST32_ABS_LO12_NC
    286: ("lo12", 3, 12, 10),  # LDST64_ABS_LO12_NC
    299: ("lo12", 4, 12, 10),  # LDST128_ABS_LO12_NC
}
for _t in _A64:
    _RELOCS[(_EM_AARCH64, _t)] = "a64"


def _a64_patch(insn: int, r_type: int, S: int, A: int, P: int, name: str) -> int:
    kind, shift, width, lsb = _A64[r_type]
    if kind == "lo12":
        v = (S + A) & 0xFFF
    else:
        v = ((S + A) & ~0xFFF) - (P & ~0xFFF) if kind == "page" else S + A - P
        limit = 1 << (width + shift - 1)
        if v & ((1 << shift) - 1) or not -limit <= v < limit:
            raise AsmError(f"'{name}' out of range or misaligned for relocation {r_type}")
    imm = (v >> shift) & ((1 << width) - 1)
    if lsb is None:  # adr/adrp: immlo[30:29], immhi[23:5]
        return (insn & ~0x60FFFFE0 & 0xFFFFFFFF) | ((imm & 3) << 29) | ((imm >> 2) << 5)
    mask = ((1 << width) - 1) << lsb
    return (insn & ~mask & 0xFFFFFFFF) | (imm << lsb)


def _elf_endian(elf: bytes) -> str:
    """Return struct format prefix based on ELF EI_DATA byte."""
    ei_data = elf[5]
    if ei_data == 1: return "<"
    if ei_data == 2: return ">"
    raise AsmError(f"unsupported ELF endianness (EI_DATA={ei_data})")


def extract_text(obj: bytes, base: int = 0) -> bytes:
    """Extract the code section from an ELF, Mach-O, or COFF object.

    For ELF, applies the relocations LLVM left in .text as if the section
    were loaded at *base*, and raises on undefined symbols or relocation
    types it cannot resolve.  Mach-O/COFF are a fallback: no relocations.
    """
    if len(obj) < 4:
        raise AsmError("backend did not produce valid object output")
    magic = obj[:4]
    if magic == _ELF_MAGIC:
        if len(obj) < 6:
            raise AsmError("truncated ELF output")
        endian = _elf_endian(obj)
        ei_class = obj[4]
        if ei_class in (1, 2): return _extract_text_elf(obj, endian, ei_class == 2, base)
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


# -- ELF ------------------------------------------------------------------

def _extract_text_elf(elf: bytes, e: str, is64: bool, base: int) -> bytes:
    if is64:
        e_shoff = struct.unpack_from(f"{e}Q", elf, 0x28)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(f"{e}HHH", elf, 0x3A)
        shdr_struct = struct.Struct(f"{e}IIQQQQIIQQ")
    else:
        e_shoff = struct.unpack_from(f"{e}I", elf, 0x20)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(f"{e}HHH", elf, 0x2E)
        shdr_struct = struct.Struct(f"{e}IIIIIIIIII")
    # Section header fields: 0=name 1=type 2=flags 3=addr 4=offset 5=size
    #                        6=link 7=info 8=addralign 9=entsize
    shdrs = [shdr_struct.unpack_from(elf, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    strtab_shdr = shdrs[e_shstrndx]
    strtab = elf[strtab_shdr[4]:strtab_shdr[4] + strtab_shdr[5]]

    for text_idx, shdr in enumerate(shdrs):
        if strtab[shdr[0]:shdr[0] + 6] == b".text\x00":
            text_bytes = bytearray(elf[shdr[4]:shdr[4] + shdr[5]])
            break
    else:
        raise AsmError("no .text section in ELF output")

    # Relocation sections targeting .text (sh_info == .text index)
    machine = struct.unpack_from(f"{e}H", elf, 0x12)[0]
    for shdr in shdrs:
        if shdr[1] in (_SHT_REL, _SHT_RELA) and shdr[7] == text_idx:
            _apply_relocs(elf, e, is64, machine, shdr, shdrs, text_idx, text_bytes, base)

    return bytes(text_bytes)


def _apply_relocs(elf, e, is64, machine, rel, shdrs, text_idx, text, base):
    """Resolve relocations in *text* as if .text were loaded at *base*."""
    rela = rel[1] == _SHT_RELA
    symtab = shdrs[rel[6]]
    symstr = shdrs[symtab[6]]
    ent = rel[9]
    if is64:
        rfmt, sfmt = f"{e}QQq" if rela else f"{e}QQ", f"{e}IBBHQ"
    else:
        rfmt, sfmt = f"{e}IIi" if rela else f"{e}II", f"{e}IIIBBH"

    for j in range(rel[5] // ent if ent else 0):
        r = struct.unpack_from(rfmt, elf, rel[4] + j * ent)
        r_offset, r_info = r[0], r[1]
        r_type, r_sym = (r_info & 0xFFFFFFFF, r_info >> 32) if is64 else (r_info & 0xFF, r_info >> 8)

        kind = _RELOCS.get((machine, r_type))
        if kind == "ignore":
            continue

        name, S = "", 0
        if r_sym:
            sym = struct.unpack_from(sfmt, elf, symtab[4] + r_sym * symtab[9])
            st_name, value, shndx = (sym[0], sym[4], sym[3]) if is64 else (sym[0], sym[1], sym[5])
            start = symstr[4] + st_name
            name = elf[start:elf.index(b"\x00", start)].decode(errors="replace")
            if shndx == _SHN_UNDEF:
                raise AsmError(f"undefined symbol: {name}")
            if shndx == _SHN_ABS:
                S = value
            elif shndx == text_idx:
                S = base + value
            else:
                raise AsmError(f"reference to '{name}' in another section is not supported")

        if kind is None:
            raise AsmError(
                f"cannot resolve relocation type {r_type} (machine {machine}) "
                f"at offset {r_offset}" + (f" for '{name}'" if name else "")
            )

        P = base + r_offset
        if kind == "abs64":
            A = r[2] if rela else struct.unpack_from(f"{e}q", text, r_offset)[0]
            struct.pack_into(f"{e}Q", text, r_offset, (S + A) & 0xFFFFFFFFFFFFFFFF)
            continue
        A = r[2] if rela else struct.unpack_from(f"{e}i", text, r_offset)[0]
        if kind == "abs32":
            v = S + A
            if not -2**31 <= v < 2**32:
                raise AsmError(f"value 0x{v:x} for '{name}' does not fit in 32 bits")
            struct.pack_into(f"{e}I", text, r_offset, v & 0xFFFFFFFF)
        elif kind == "pc32":
            struct.pack_into(f"{e}I", text, r_offset, (S + A - P) & 0xFFFFFFFF)
        elif kind == "a64":
            insn = struct.unpack_from("<I", text, r_offset)[0]  # A64 code is always little-endian
            struct.pack_into("<I", text, r_offset, _a64_patch(insn, r_type, S, A, P, name))


# -- Mach-O ----------------------------------------------------------------

def _extract_text_macho64(obj: bytes) -> bytes:
    """Extract __text section from a 64-bit Mach-O object."""
    ncmds = struct.unpack_from("<I", obj, 16)[0]
    off = 32  # past mach_header_64
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        if cmd == _LC_SEGMENT_64:
            nsects = struct.unpack_from("<I", obj, off + 64)[0]
            sect_off = off + 72
            for _ in range(nsects):
                sectname = obj[sect_off:sect_off + 16].rstrip(b"\x00")
                if sectname == b"__text":
                    size = struct.unpack_from("<Q", obj, sect_off + 40)[0]
                    offset = struct.unpack_from("<I", obj, sect_off + 48)[0]
                    return obj[offset:offset + size]
                sect_off += 80
        off += cmdsize
    raise AsmError("no __text section in Mach-O output")


def _extract_text_macho32(obj: bytes) -> bytes:
    """Extract __text section from a 32-bit Mach-O object."""
    ncmds = struct.unpack_from("<I", obj, 16)[0]
    off = 28
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        if cmd == _LC_SEGMENT:
            nsects = struct.unpack_from("<I", obj, off + 48)[0]
            sect_off = off + 56
            for _ in range(nsects):
                sectname = obj[sect_off:sect_off + 16].rstrip(b"\x00")
                if sectname == b"__text":
                    size = struct.unpack_from("<I", obj, sect_off + 36)[0]
                    offset = struct.unpack_from("<I", obj, sect_off + 40)[0]
                    return obj[offset:offset + size]
                sect_off += 68
        off += cmdsize
    raise AsmError("no __text section in Mach-O output")


# -- COFF ------------------------------------------------------------------

def _extract_text_coff(obj: bytes) -> bytes:
    """Extract .text section from a COFF object."""
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
