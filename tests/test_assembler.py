"""Tests for flint_mc assembler."""

import pytest

from flint_mc import Assembler, AsmError


@pytest.fixture(scope="module")
def asm():
    return Assembler()


# ---------------------------------------------------------------------------
# Basic Thumb-2
# ---------------------------------------------------------------------------

class TestBasicThumb:
    def test_nop(self, asm):
        assert asm.asm("nop") == b"\x00\xbf"

    def test_mov_immediate(self, asm):
        code = asm.asm("mov r0, #0")
        assert len(code) in (2, 4)

    def test_bx_lr(self, asm):
        assert asm.asm("bx lr") == b"\x70\x47"

    def test_multi_instructions(self, asm):
        code = asm.asm("mov r0, #0; bx lr")
        assert len(code) >= 4  # at least 2 instructions

    def test_multi_newlines(self, asm):
        code = asm.asm("mov r0, #0\nbx lr")
        assert len(code) >= 4

    def test_asm_one(self, asm):
        code = asm.asm_one("nop")
        assert code == b"\x00\xbf"

    def test_asm_one_rejects_multi(self, asm):
        with pytest.raises(AsmError, match="single instruction"):
            asm.asm_one("nop; nop; nop")


# ---------------------------------------------------------------------------
# Cortex-M special registers (the keystone gaps)
# ---------------------------------------------------------------------------

class TestMClassRegisters:
    def test_mrs_primask(self, asm):
        code = asm.asm("mrs r0, PRIMASK")
        assert code == bytes.fromhex("eff31080")

    def test_msr_basepri(self, asm):
        code = asm.asm("msr BASEPRI, r1")
        assert code == bytes.fromhex("81f31188")

    def test_mrs_faultmask(self, asm):
        code = asm.asm("mrs r0, FAULTMASK")
        assert len(code) == 4

    def test_msr_control(self, asm):
        code = asm.asm("msr CONTROL, r0")
        assert len(code) == 4

    def test_mrs_basepri_max(self, asm):
        code = asm.asm("mrs r0, BASEPRI_MAX")
        assert len(code) == 4


# ---------------------------------------------------------------------------
# FPv5 instructions (also missing from keystone)
# ---------------------------------------------------------------------------

class TestFPv5:
    def test_vfma(self, asm):
        code = asm.asm("vfma.f32 s0, s1, s2")
        assert code == bytes.fromhex("a0ee810a")

    def test_vfms(self, asm):
        code = asm.asm("vfms.f32 s0, s1, s2")
        assert len(code) == 4

    def test_vfnma(self, asm):
        code = asm.asm("vfnma.f32 s0, s1, s2")
        assert len(code) == 4

    def test_vmov_f32_imm(self, asm):
        code = asm.asm("vmov.f32 s0, #1.0")
        assert len(code) == 4

    def test_vldr_vstr(self, asm):
        code = asm.asm("vldr s0, [r0, #0]\nvstr s0, [r1, #4]")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Address handling
# ---------------------------------------------------------------------------

class TestAddressing:
    def test_addr_zero_default(self, asm):
        code = asm.asm("nop")
        assert code == asm.asm("nop", addr=0)

    def test_branch_with_addr(self, asm):
        # A branch to itself at address 0x1000
        code = asm.asm("b .", addr=0x1000)
        assert len(code) in (2, 4)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_instruction(self, asm):
        with pytest.raises(AsmError):
            asm.asm("not_a_real_instruction r0, r1")

    def test_bad_register(self, asm):
        with pytest.raises(AsmError):
            asm.asm("mov r99, #0")

    def test_bad_llvm_path(self):
        with pytest.raises(FileNotFoundError):
            Assembler(llvm_mc="/nonexistent/llvm-mc")

    def test_empty_input(self, asm):
        # Empty asm should produce empty .text or raise
        code = asm.asm("")
        assert code == b""
