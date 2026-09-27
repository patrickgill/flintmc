"""ELF relocation handling in objfile.extract_text.

LLVM leaves relocations in .text for undefined symbols, global labels,
AArch64 adrp/:lo12: pairs and absolute data references.  Each must either
be applied or raise -- never ship an unpatched (silently wrong) field.
"""

import pytest
from flintmc import Assembler, AsmError

pytestmark = pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")


@pytest.mark.parametrize("profile, source", [
    ("x86_64", "call prinft"),
    ("x86_64", "movabs rax, missing"),
    ("aarch64", "bl prinft"),
    ("aarch64", "ldr x0, =missing"),
    ("cortex_m4", "bl prinft"),
    ("cortex_m4", "ldr r0, =missing"),
    ("riscv64", "call prinft"),
])
def test_undefined_symbol_raises(profile, source):
    with pytest.raises(AsmError, match=r"undefined symbol: (prinft|missing)"):
        getattr(Assembler, profile)()(source)


@pytest.mark.parametrize("source", [
    "cbz x0, g\ntbz x0, #1, g\nb.eq g\nadr x1, g\nldr x2, g\nbl g\nb g\ng: nop",
    "adrp x0, g\nadd x0, x0, :lo12:g\n.space 0x3000\ng: nop",
])
def test_aarch64_global_label_matches_local(source):
    # A .global label forces relocations; the result must equal the local-label encoding.
    asm = Assembler.aarch64()
    assert asm(".global g\n" + source) == asm(source)


@pytest.mark.parametrize("address, page, lo12", [
    # t is at block+0x1ff8; adrp page delta and :lo12: depend on where the block sits
    (0, "#4096", "#0xff8"),
    (0x0F_0800, "#8192", "#0x7f8"),
])
def test_aarch64_adrp_lo12_uses_address(address, page, lo12):
    from flintmc import Disassembler
    code = Assembler.aarch64().asm("adrp x0, t\nldr x1, [x0, :lo12:t]\n.space 0x1ff0\nt: .quad 1",
                                   address=address)
    adrp, ldr = (i.text.replace("\t", " ") for i in Disassembler.aarch64()(code[:8]))
    assert adrp == f"adrp x0, {page}"
    assert ldr == f"ldr x1, [x0, {lo12}]"


def test_absolute_label_reference_x86_64():
    code = Assembler("x86_64", preamble="").asm("movabsq $t, %rax\nt: ret", address=0x1000)
    assert code[2:10] == (0x1000 + 10).to_bytes(8, "little")


def test_absolute_label_reference_aarch64_literal():
    code = Assembler.aarch64().asm("ldr x0, =t\nt: ret", address=0x2000)
    # literal pool (after ret, 8-byte aligned) holds t's absolute address
    assert (0x2000 + 4).to_bytes(8, "little") in code


def test_unsupported_relocation_raises():
    # RISC-V call to an absolute .set symbol needs R_RISCV_CALL_PLT: refuse, don't emit call-to-self
    with pytest.raises(AsmError, match="cannot resolve relocation"):
        Assembler.riscv64()("call h", symbols={"h": 0x2000})
