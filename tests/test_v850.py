"""Tests for V850 / V850ES / V850E1 via the GNU binutils backend."""

import pytest
from flintmc import Assembler, Disassembler, AsmError
from flintmc.binutils import find_tool

pytestmark = pytest.mark.skipif(
    not (find_tool("v850-elf", "as") and find_tool("v850-elf", "objdump")),
    reason="v850-elf binutils not found (set BINUTILS_PATH)",
)


@pytest.fixture(scope="module", params=["v850", "v850es", "v850e1"])
def asm(request):
    return getattr(Assembler, request.param)()


def test_backend_auto(asm):
    assert asm.backend == "binutils"


def test_basic(asm):
    assert asm("nop") == b"\x00\x00"
    assert asm("nop; mov 5, r1") == b"\x00\x00\x05\x0a"


def test_e1_only_instruction():
    assert len(Assembler.v850e1()("mul r1, r2, r3")) == 4
    assert Assembler.v850es()("mul r1, r2, r3") == Assembler.v850e1()("mul r1, r2, r3")
    with pytest.raises(AsmError, match="does not support"):
        Assembler.v850()("mul r1, r2, r3")


def test_error_line_number(asm):
    with pytest.raises(AsmError, match="line 2:.*bogus"):
        asm("nop\nbogus r1")


def test_asm_each(asm):
    each = asm.asm_each("nop; mov 5, r1; jr 0")
    assert [(i.offset, i.size, i.source) for i in each] == [
        (0, 2, "nop"), (2, 2, "mov 5, r1"), (4, 4, "jr 0")]


def test_labels_and_address(asm):
    code = asm("loop: add -1, r1; bne loop")
    assert len(code) == 4
    assert asm("nop", address=0x100) == b"\x00\x00"


def test_round_trip(asm):
    src = "nop; mov 5, r1; add r1, r2; ld.w 4[r3], r4"
    code = asm(src)
    dis = Disassembler(asm.triple, cpu=asm.cpu)
    out = dis(code)
    assert [i.mnemonic for i in out] == ["nop", "mov", "add", "ld.w"]
    assert asm("\n".join(i.text for i in out)) == code
    assert asm(src, verify=True) == code


def test_disasm_fields_and_count():
    out = Disassembler.v850e1()(b"\x00\x00\x05\x0a", address=0x1000, count=1)
    assert len(out) == 1
    assert (out[0].offset, out[0].size, out[0].address, out[0].text) == (0, 2, 0x1000, "nop")


def test_branch_displacement_round_trip():
    # jr/jarl/bcond from SP-303 firmware: objdump prints absolute targets,
    # but gas takes displacements, so the text must be a displacement.
    code = bytes.fromhex("8507c0bf" "a82f1000" "e2fd" "0000")
    out = Disassembler.v850e1()(code, address=0x100140)
    assert [i.text for i in out[:3]] == ["jr 0x5bfc0", "jarl -0x17fff0, r5", "be -0x4"]
    asm = Assembler.v850e1()
    assert asm("\n".join(i.text for i in out), address=0x100140) == code
    assert asm("jr 0x5bfc0; nop", verify=True) == bytes.fromhex("8507c0bf0000")


def test_objdump_quirks_round_trip():
    # From SP-303 firmware: aliased sysreg/condition names, dispose without jump
    code = bytes.fromhex("e00f4000" "e18f2000" "e9370000" "60060000")
    out = Disassembler.v850e1()(code, strict=True)
    assert [i.text for i in out] == ["stsr eipc, r1", "ldsr r1, ctpsw", "setf nc, r6", "dispose 16, {}"]
    assert Assembler.v850e1()("\n".join(i.text for i in out)) == code


def test_extension_ops_need_mextension():
    code = bytes.fromhex("ee77f279")  # sdivun: optional extension, gas needs -mextension
    with pytest.raises(AsmError):
        Disassembler.v850e1()(code, strict=True)
    with pytest.raises(AsmError):
        Assembler.v850e1()("sdivun 8, r14, r14, r15")
    ext = dict(features="-mextension")
    out = Disassembler.v850e1(**ext)(code, strict=True)
    assert out[0].text == "sdivun 8, r14, r14, r15"
    assert Assembler.v850e1(**ext)(out[0].text) == code
