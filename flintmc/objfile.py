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

# ELF section types
_SHT_RELA = 4
_SHT_SYMTAB = 2

# x86_64 relocation types — PC-relative 32-bit
_R_X86_64_PC32 = 2
_R_X86_64_PLT32 = 4


def _elf_endian(elf: bytes) -> str:
    """Return struct format prefix based on ELF EI_DATA byte."""
    ei_data = elf[5]
    if ei_data == 1: return "<"
    if ei_data == 2: return ">"
    raise AsmError(f"unsupported ELF endianness (EI_DATA={ei_data})")


def extract_text(obj: bytes) -> bytes:
    """Extract the code section from an ELF, Mach-O, or COFF object.

    For ELF64, applies .rela.text relocations (R_X86_64_PC32 etc.)
    so that branch instructions to .set symbols resolve correctly.
    """
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


# -- ELF ------------------------------------------------------------------

_ELF32_SHDR = struct.Struct("<IIIIIIIIII")
_ELF64_SHDR = struct.Struct("<IIQQQQIIQQ")
# Indices into unpacked section header:
#   0=name 1=type 2=flags 3=addr 4=offset 5=size 6=link 7=info 8=addralign 9=entsize


def _extract_text_elf32(elf: bytes, e: str) -> bytes:
    e_shoff = struct.unpack_from(f"{e}I", elf, 0x20)[0]
    e_shentsize = struct.unpack_from(f"{e}H", elf, 0x2E)[0]
    e_shnum = struct.unpack_from(f"{e}H", elf, 0x30)[0]
    e_shstrndx = struct.unpack_from(f"{e}H", elf, 0x32)[0]
    shdr_struct = struct.Struct(f"{e}IIIIIIIIII")
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, shdr_struct, 4, 5, e)


def _extract_text_elf64(elf: bytes, e: str) -> bytes:
    e_shoff = struct.unpack_from(f"{e}Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from(f"{e}H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from(f"{e}H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from(f"{e}H", elf, 0x3E)[0]
    shdr_struct = struct.Struct(f"{e}IIQQQQIIQQ")

    # Find section name string table
    strtab_shdr = shdr_struct.unpack_from(elf, e_shoff + e_shstrndx * e_shentsize)
    strtab = elf[strtab_shdr[4]:strtab_shdr[4] + strtab_shdr[5]]

    # Locate .text section and its index
    text_bytes = None
    text_idx = None
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name_off = struct.unpack_from(f"{e}I", elf, off)[0]
        if strtab[name_off:name_off + 6] == b".text\x00":
            shdr = shdr_struct.unpack_from(elf, off)
            text_bytes = bytearray(elf[shdr[4]:shdr[4] + shdr[5]])
            text_idx = i
            break

    if text_bytes is None:
        raise AsmError("no .text section in ELF output")

    # Apply relocations from .rela.text if present
    for i in range(e_shnum):
        shdr = shdr_struct.unpack_from(elf, e_shoff + i * e_shentsize)
        # sh_type == SHT_RELA and sh_info == .text section index
        if shdr[1] == _SHT_RELA and shdr[7] == text_idx:
            _apply_elf64_rela(elf, e, shdr, shdr_struct, e_shoff, e_shentsize, text_bytes)
            break

    return bytes(text_bytes)


def _apply_elf64_rela(elf, e, rela_shdr, shdr_struct, e_shoff, e_shentsize, text):
    """Apply ELF64 RELA relocations to .text bytes."""
    rela_off = rela_shdr[4]
    rela_size = rela_shdr[5]
    rela_entsize = rela_shdr[9]
    if rela_entsize == 0:
        return

    # Load symbol table (rela sh_link points to it)
    symtab_shdr = shdr_struct.unpack_from(elf, e_shoff + rela_shdr[6] * e_shentsize)
    sym_entsize = symtab_shdr[9]

    n_rela = rela_size // rela_entsize
    for j in range(n_rela):
        off = rela_off + j * rela_entsize
        r_offset, r_info, r_addend = struct.unpack_from(f"{e}QQq", elf, off)
        r_type = r_info & 0xFFFFFFFF
        r_sym_idx = r_info >> 32

        # Resolve symbol value
        if r_sym_idx != 0 and sym_entsize:
            sym_off = symtab_shdr[4] + r_sym_idx * sym_entsize
            # Elf64_Sym: st_name(I) st_info(B) st_other(B) st_shndx(H) st_value(Q) st_size(Q)
            st_value = struct.unpack_from(f"{e}Q", elf, sym_off + 8)[0]
        else:
            st_value = 0

        # Apply relocation
        if r_type in (_R_X86_64_PC32, _R_X86_64_PLT32):
            # S + A - P (32-bit PC-relative)
            # P = r_offset (section is at address 0 in relocatable objects)
            value = (st_value + r_addend - r_offset) & 0xFFFFFFFF
            struct.pack_into("<I", text, r_offset, value)
        # Other relocation types: leave as-is (already resolved or unsupported)


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
