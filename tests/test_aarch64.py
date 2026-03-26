"""Tests for AArch64 (ARMv8-A 64-bit) assembly."""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def asm():
    return Assembler.aarch64()


# ---------------------------------------------------------------------------
# Basic instructions
# ---------------------------------------------------------------------------

class TestBasic:
    def test_nop(self, asm):
        assert asm.asm("nop") == b"\x1f\x20\x03\xd5"

    def test_ret(self, asm):
        assert asm.asm("ret") == b"\xc0\x03\x5f\xd6"

    def test_mov_immediate(self, asm):
        code = asm.asm("mov x0, #42")
        assert len(code) == 4

    def test_mov_register(self, asm):
        code = asm.asm("mov x0, x1")
        assert len(code) == 4

    def test_mov_w_register(self, asm):
        code = asm.asm("mov w0, w1")
        assert len(code) == 4

    def test_multi_instructions(self, asm):
        code = asm.asm("mov x0, #0; ret")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add_reg(self, asm):
        code = asm.asm("add x0, x1, x2")
        assert len(code) == 4

    def test_add_imm(self, asm):
        code = asm.asm("add x0, x1, #1")
        assert len(code) == 4

    def test_sub_reg(self, asm):
        code = asm.asm("sub x0, x1, x2")
        assert len(code) == 4

    def test_adds_sets_flags(self, asm):
        code = asm.asm("adds x0, x1, x2")
        assert len(code) == 4

    def test_mul(self, asm):
        code = asm.asm("mul x0, x1, x2")
        assert len(code) == 4

    def test_madd(self, asm):
        code = asm.asm("madd x0, x1, x2, x3")
        assert len(code) == 4

    def test_sdiv(self, asm):
        code = asm.asm("sdiv x0, x1, x2")
        assert len(code) == 4

    def test_udiv(self, asm):
        code = asm.asm("udiv x0, x1, x2")
        assert len(code) == 4

    def test_neg(self, asm):
        code = asm.asm("neg x0, x1")
        assert len(code) == 4

    def test_32bit_arithmetic(self, asm):
        code = asm.asm("add w0, w1, w2")
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Logic and shifts
# ---------------------------------------------------------------------------

class TestLogic:
    def test_and(self, asm):
        assert len(asm.asm("and x0, x1, x2")) == 4

    def test_orr(self, asm):
        assert len(asm.asm("orr x0, x1, x2")) == 4

    def test_eor(self, asm):
        assert len(asm.asm("eor x0, x1, x2")) == 4

    def test_bic(self, asm):
        assert len(asm.asm("bic x0, x1, x2")) == 4

    def test_lsl_imm(self, asm):
        assert len(asm.asm("lsl x0, x1, #4")) == 4

    def test_lsr_imm(self, asm):
        assert len(asm.asm("lsr x0, x1, #4")) == 4

    def test_asr_imm(self, asm):
        assert len(asm.asm("asr x0, x1, #4")) == 4

    def test_ror_reg(self, asm):
        assert len(asm.asm("ror x0, x1, x2")) == 4

    def test_clz(self, asm):
        assert len(asm.asm("clz x0, x1")) == 4

    def test_rbit(self, asm):
        assert len(asm.asm("rbit x0, x1")) == 4

    def test_rev(self, asm):
        assert len(asm.asm("rev x0, x1")) == 4


# ---------------------------------------------------------------------------
# Load / store
# ---------------------------------------------------------------------------

class TestLoadStore:
    def test_ldr_reg_offset(self, asm):
        assert len(asm.asm("ldr x0, [x1]")) == 4

    def test_ldr_imm_offset(self, asm):
        assert len(asm.asm("ldr x0, [x1, #8]")) == 4

    def test_str_reg(self, asm):
        assert len(asm.asm("str x0, [x1]")) == 4

    def test_ldr_pre_index(self, asm):
        assert len(asm.asm("ldr x0, [x1, #16]!")) == 4

    def test_ldr_post_index(self, asm):
        assert len(asm.asm("ldr x0, [x1], #16")) == 4

    def test_ldp(self, asm):
        assert len(asm.asm("ldp x0, x1, [sp]")) == 4

    def test_stp(self, asm):
        assert len(asm.asm("stp x0, x1, [sp, #-16]!")) == 4

    def test_ldrb(self, asm):
        assert len(asm.asm("ldrb w0, [x1]")) == 4

    def test_ldrh(self, asm):
        assert len(asm.asm("ldrh w0, [x1]")) == 4

    def test_ldrsb(self, asm):
        assert len(asm.asm("ldrsb x0, [x1]")) == 4

    def test_ldrsh(self, asm):
        assert len(asm.asm("ldrsh x0, [x1]")) == 4

    def test_ldrsw(self, asm):
        assert len(asm.asm("ldrsw x0, [x1]")) == 4

    def test_ldr_register_offset(self, asm):
        assert len(asm.asm("ldr x0, [x1, x2]")) == 4

    def test_ldr_shifted_register(self, asm):
        assert len(asm.asm("ldr x0, [x1, x2, lsl #3]")) == 4


# ---------------------------------------------------------------------------
# Branches and control flow
# ---------------------------------------------------------------------------

class TestBranches:
    def test_b_label(self, asm):
        code = asm.asm("b target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_bl_label(self, asm):
        code = asm.asm("bl target\ntarget:\nnop")
        assert len(code) == 8

    def test_br(self, asm):
        assert len(asm.asm("br x0")) == 4

    def test_blr(self, asm):
        assert len(asm.asm("blr x0")) == 4

    def test_cbz(self, asm):
        code = asm.asm("cbz x0, skip\nnop\nskip:\nnop")
        assert len(code) == 12

    def test_cbnz(self, asm):
        code = asm.asm("cbnz x0, skip\nnop\nskip:\nnop")
        assert len(code) == 12

    def test_tbz(self, asm):
        code = asm.asm("tbz x0, #0, skip\nnop\nskip:\nnop")
        assert len(code) == 12

    def test_tbnz(self, asm):
        code = asm.asm("tbnz x0, #0, skip\nnop\nskip:\nnop")
        assert len(code) == 12

    def test_conditional_branch(self, asm):
        code = asm.asm("cmp x0, #0\nb.eq target\nnop\ntarget:\nnop")
        assert len(code) == 16

    def test_backward_branch(self, asm):
        code = asm.asm("loop:\nnop\nb loop")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Conditional select
# ---------------------------------------------------------------------------

class TestConditionalSelect:
    def test_csel(self, asm):
        assert len(asm.asm("csel x0, x1, x2, eq")) == 4

    def test_csinc(self, asm):
        assert len(asm.asm("csinc x0, x1, x2, ne")) == 4

    def test_cset(self, asm):
        assert len(asm.asm("cset x0, eq")) == 4

    def test_csinv(self, asm):
        assert len(asm.asm("csinv x0, x1, x2, lt")) == 4


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

class TestComparisons:
    def test_cmp_imm(self, asm):
        assert len(asm.asm("cmp x0, #0")) == 4

    def test_cmp_reg(self, asm):
        assert len(asm.asm("cmp x0, x1")) == 4

    def test_cmn(self, asm):
        assert len(asm.asm("cmn x0, x1")) == 4

    def test_tst_imm(self, asm):
        assert len(asm.asm("tst x0, #0xff")) == 4

    def test_tst_reg(self, asm):
        assert len(asm.asm("tst x0, x1")) == 4


# ---------------------------------------------------------------------------
# Move wide / bitfield
# ---------------------------------------------------------------------------

class TestMoveWide:
    def test_movz(self, asm):
        assert len(asm.asm("movz x0, #0x1234")) == 4

    def test_movk(self, asm):
        assert len(asm.asm("movk x0, #0x5678, lsl #16")) == 4

    def test_movn(self, asm):
        assert len(asm.asm("movn x0, #0")) == 4

    def test_bfm(self, asm):
        assert len(asm.asm("ubfx x0, x1, #0, #8")) == 4

    def test_bfi(self, asm):
        assert len(asm.asm("bfi x0, x1, #8, #4")) == 4

    def test_sbfx(self, asm):
        assert len(asm.asm("sbfx x0, x1, #0, #16")) == 4

    def test_extr(self, asm):
        assert len(asm.asm("extr x0, x1, x2, #4")) == 4


# ---------------------------------------------------------------------------
# SIMD / NEON
# ---------------------------------------------------------------------------

class TestSIMD:
    def test_fmov_imm(self, asm):
        assert len(asm.asm("fmov d0, #1.0")) == 4

    def test_fmov_reg(self, asm):
        assert len(asm.asm("fmov d0, d1")) == 4

    def test_fadd_scalar(self, asm):
        assert len(asm.asm("fadd d0, d1, d2")) == 4

    def test_fsub_scalar(self, asm):
        assert len(asm.asm("fsub d0, d1, d2")) == 4

    def test_fmul_scalar(self, asm):
        assert len(asm.asm("fmul d0, d1, d2")) == 4

    def test_fdiv_scalar(self, asm):
        assert len(asm.asm("fdiv d0, d1, d2")) == 4

    def test_fmadd(self, asm):
        assert len(asm.asm("fmadd d0, d1, d2, d3")) == 4

    def test_fcmp(self, asm):
        assert len(asm.asm("fcmp d0, d1")) == 4

    def test_fcvtzs(self, asm):
        assert len(asm.asm("fcvtzs x0, d0")) == 4

    def test_scvtf(self, asm):
        assert len(asm.asm("scvtf d0, x0")) == 4

    def test_add_vector(self, asm):
        assert len(asm.asm("add v0.4s, v1.4s, v2.4s")) == 4

    def test_fadd_vector(self, asm):
        assert len(asm.asm("fadd v0.4s, v1.4s, v2.4s")) == 4

    def test_fmul_vector(self, asm):
        assert len(asm.asm("fmul v0.4s, v1.4s, v2.4s")) == 4

    def test_dup_element(self, asm):
        assert len(asm.asm("dup v0.4s, v1.s[0]")) == 4

    def test_ins_element(self, asm):
        assert len(asm.asm("ins v0.s[0], w0")) == 4

    def test_umov(self, asm):
        assert len(asm.asm("umov w0, v0.s[0]")) == 4

    def test_movi(self, asm):
        assert len(asm.asm("movi v0.4s, #0")) == 4

    def test_ldr_q(self, asm):
        assert len(asm.asm("ldr q0, [x0]")) == 4

    def test_str_q(self, asm):
        assert len(asm.asm("str q0, [x0]")) == 4

    def test_ldp_q(self, asm):
        assert len(asm.asm("ldp q0, q1, [x0]")) == 4

    def test_single_precision(self, asm):
        assert len(asm.asm("fadd s0, s1, s2")) == 4

    def test_half_precision(self, asm):
        assert len(asm.asm("fadd h0, h1, h2")) == 4


# ---------------------------------------------------------------------------
# System / barriers / hints
# ---------------------------------------------------------------------------

class TestSystem:
    def test_svc(self, asm):
        assert len(asm.asm("svc #0")) == 4

    def test_brk(self, asm):
        assert len(asm.asm("brk #0")) == 4

    def test_hlt(self, asm):
        assert len(asm.asm("hlt #0")) == 4

    def test_dmb(self, asm):
        assert len(asm.asm("dmb sy")) == 4

    def test_dsb(self, asm):
        assert len(asm.asm("dsb sy")) == 4

    def test_isb(self, asm):
        assert len(asm.asm("isb")) == 4

    def test_wfi(self, asm):
        assert len(asm.asm("wfi")) == 4

    def test_wfe(self, asm):
        assert len(asm.asm("wfe")) == 4

    def test_mrs(self, asm):
        assert len(asm.asm("mrs x0, NZCV")) == 4

    def test_msr(self, asm):
        assert len(asm.asm("msr NZCV, x0")) == 4


# ---------------------------------------------------------------------------
# Stack operations (function prologue/epilogue patterns)
# ---------------------------------------------------------------------------

class TestStackPatterns:
    def test_prologue(self, asm):
        code = asm.asm("stp x29, x30, [sp, #-16]!\nmov x29, sp")
        assert len(code) == 8

    def test_epilogue(self, asm):
        code = asm.asm("ldp x29, x30, [sp], #16\nret")
        assert len(code) == 8

    def test_callee_saved(self, asm):
        code = asm.asm("""
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
        """)
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------

class TestAtomics:
    def test_ldxr(self, asm):
        assert len(asm.asm("ldxr x0, [x1]")) == 4

    def test_stxr(self, asm):
        assert len(asm.asm("stxr w0, x1, [x2]")) == 4

    def test_ldaxr(self, asm):
        assert len(asm.asm("ldaxr x0, [x1]")) == 4

    def test_stlxr(self, asm):
        assert len(asm.asm("stlxr w0, x1, [x2]")) == 4

    def test_cas(self, asm):
        """Compare-and-swap (ARMv8.1-A LSE)."""
        a = Assembler(triple="aarch64", features="+lse")
        assert len(a.asm("cas x0, x1, [x2]")) == 4

    def test_ldadd(self, asm):
        """Atomic add (ARMv8.1-A LSE)."""
        a = Assembler(triple="aarch64", features="+lse")
        assert len(a.asm("ldadd x0, x1, [x2]")) == 4


# ---------------------------------------------------------------------------
# Address generation
# ---------------------------------------------------------------------------

class TestAddressGen:
    def test_adr(self, asm):
        code = asm.asm("adr x0, target\ntarget:\nnop")
        assert len(code) == 8

    def test_adrp(self, asm):
        code = asm.asm("adrp x0, target\ntarget:\nnop")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_instruction(self, asm):
        with pytest.raises(AsmError):
            asm.asm("not_real x0, x1")

    def test_bad_register(self, asm):
        with pytest.raises(AsmError):
            asm.asm("mov x99, x0")

    def test_arm32_rejected(self, asm):
        """ARM32 mnemonics should fail on AArch64."""
        with pytest.raises(AsmError):
            asm.asm("push {r0}")
