"""Tests for RISC-V assembly (RV32 and RV64)."""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def rv64():
    return Assembler(
        triple="riscv64",
        cpu="generic-rv64",
        features="+m,+a,+f,+d",
    )


@pytest.fixture(scope="module")
def rv32():
    return Assembler(
        triple="riscv32",
        cpu="generic-rv32",
        features="+m,+a,+f",
    )


# ---------------------------------------------------------------------------
# RV64I — base integer
# ---------------------------------------------------------------------------

class TestRV64Base:
    def test_nop(self, rv64):
        code = rv64.asm("nop")
        assert len(code) == 4

    def test_addi(self, rv64):
        code = rv64.asm("addi x1, x0, 42")
        assert len(code) == 4

    def test_add(self, rv64):
        assert len(rv64.asm("add x1, x2, x3")) == 4

    def test_sub(self, rv64):
        assert len(rv64.asm("sub x1, x2, x3")) == 4

    def test_and(self, rv64):
        assert len(rv64.asm("and x1, x2, x3")) == 4

    def test_or(self, rv64):
        assert len(rv64.asm("or x1, x2, x3")) == 4

    def test_xor(self, rv64):
        assert len(rv64.asm("xor x1, x2, x3")) == 4

    def test_sll(self, rv64):
        assert len(rv64.asm("sll x1, x2, x3")) == 4

    def test_srl(self, rv64):
        assert len(rv64.asm("srl x1, x2, x3")) == 4

    def test_sra(self, rv64):
        assert len(rv64.asm("sra x1, x2, x3")) == 4

    def test_slt(self, rv64):
        assert len(rv64.asm("slt x1, x2, x3")) == 4

    def test_sltu(self, rv64):
        assert len(rv64.asm("sltu x1, x2, x3")) == 4

    def test_lui(self, rv64):
        assert len(rv64.asm("lui x1, 0x12345")) == 4

    def test_auipc(self, rv64):
        assert len(rv64.asm("auipc x1, 0")) == 4

    def test_jal(self, rv64):
        code = rv64.asm("jal x1, target\ntarget:\nnop")
        assert len(code) == 8

    def test_jalr(self, rv64):
        assert len(rv64.asm("jalr x1, x2, 0")) == 4

    def test_ret(self, rv64):
        """ret is a pseudo for jalr x0, x1, 0."""
        assert len(rv64.asm("ret")) == 4

    def test_li_pseudo(self, rv64):
        """li pseudo-instruction for loading immediates."""
        code = rv64.asm("li x1, 42")
        assert len(code) >= 4

    def test_mv_pseudo(self, rv64):
        code = rv64.asm("mv x1, x2")
        assert len(code) == 4


# ---------------------------------------------------------------------------
# RV64I — loads and stores
# ---------------------------------------------------------------------------

class TestRV64LoadStore:
    def test_ld(self, rv64):
        assert len(rv64.asm("ld x1, 0(x2)")) == 4

    def test_sd(self, rv64):
        assert len(rv64.asm("sd x1, 0(x2)")) == 4

    def test_lw(self, rv64):
        assert len(rv64.asm("lw x1, 0(x2)")) == 4

    def test_sw(self, rv64):
        assert len(rv64.asm("sw x1, 0(x2)")) == 4

    def test_lh(self, rv64):
        assert len(rv64.asm("lh x1, 0(x2)")) == 4

    def test_sh(self, rv64):
        assert len(rv64.asm("sh x1, 0(x2)")) == 4

    def test_lb(self, rv64):
        assert len(rv64.asm("lb x1, 0(x2)")) == 4

    def test_sb(self, rv64):
        assert len(rv64.asm("sb x1, 0(x2)")) == 4

    def test_lbu(self, rv64):
        assert len(rv64.asm("lbu x1, 0(x2)")) == 4

    def test_lhu(self, rv64):
        assert len(rv64.asm("lhu x1, 0(x2)")) == 4

    def test_lwu(self, rv64):
        assert len(rv64.asm("lwu x1, 0(x2)")) == 4

    def test_offset(self, rv64):
        assert len(rv64.asm("ld x1, 128(x2)")) == 4


# ---------------------------------------------------------------------------
# RV64I — branches
# ---------------------------------------------------------------------------

class TestRV64Branches:
    def test_beq(self, rv64):
        code = rv64.asm("beq x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_bne(self, rv64):
        code = rv64.asm("bne x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_blt(self, rv64):
        code = rv64.asm("blt x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_bge(self, rv64):
        code = rv64.asm("bge x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_bltu(self, rv64):
        code = rv64.asm("bltu x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_bgeu(self, rv64):
        code = rv64.asm("bgeu x1, x2, target\nnop\ntarget:\nnop")
        assert len(code) == 12

    def test_backward_branch(self, rv64):
        code = rv64.asm("loop:\nnop\nbne x1, x0, loop")
        assert len(code) == 8

    def test_beqz_pseudo(self, rv64):
        code = rv64.asm("beqz x1, target\ntarget:\nnop")
        assert len(code) == 8

    def test_bnez_pseudo(self, rv64):
        code = rv64.asm("bnez x1, target\ntarget:\nnop")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# M extension — multiply / divide
# ---------------------------------------------------------------------------

class TestRV64M:
    def test_mul(self, rv64):
        assert len(rv64.asm("mul x1, x2, x3")) == 4

    def test_mulh(self, rv64):
        assert len(rv64.asm("mulh x1, x2, x3")) == 4

    def test_div(self, rv64):
        assert len(rv64.asm("div x1, x2, x3")) == 4

    def test_divu(self, rv64):
        assert len(rv64.asm("divu x1, x2, x3")) == 4

    def test_rem(self, rv64):
        assert len(rv64.asm("rem x1, x2, x3")) == 4

    def test_remu(self, rv64):
        assert len(rv64.asm("remu x1, x2, x3")) == 4

    def test_mulw(self, rv64):
        assert len(rv64.asm("mulw x1, x2, x3")) == 4

    def test_divw(self, rv64):
        assert len(rv64.asm("divw x1, x2, x3")) == 4


# ---------------------------------------------------------------------------
# A extension — atomics
# ---------------------------------------------------------------------------

class TestRV64A:
    def test_lr_sc(self, rv64):
        assert len(rv64.asm("lr.d x1, (x2)")) == 4
        assert len(rv64.asm("sc.d x1, x3, (x2)")) == 4

    def test_amoswap(self, rv64):
        assert len(rv64.asm("amoswap.d x1, x2, (x3)")) == 4

    def test_amoadd(self, rv64):
        assert len(rv64.asm("amoadd.d x1, x2, (x3)")) == 4

    def test_amoand(self, rv64):
        assert len(rv64.asm("amoand.d x1, x2, (x3)")) == 4

    def test_amoor(self, rv64):
        assert len(rv64.asm("amoor.d x1, x2, (x3)")) == 4

    def test_amomin(self, rv64):
        assert len(rv64.asm("amomin.d x1, x2, (x3)")) == 4

    def test_amomax(self, rv64):
        assert len(rv64.asm("amomax.d x1, x2, (x3)")) == 4

    def test_lr_w(self, rv64):
        assert len(rv64.asm("lr.w x1, (x2)")) == 4


# ---------------------------------------------------------------------------
# F/D extensions — floating point
# ---------------------------------------------------------------------------

class TestRV64FD:
    def test_flw(self, rv64):
        assert len(rv64.asm("flw f0, 0(x1)")) == 4

    def test_fsw(self, rv64):
        assert len(rv64.asm("fsw f0, 0(x1)")) == 4

    def test_fld(self, rv64):
        assert len(rv64.asm("fld f0, 0(x1)")) == 4

    def test_fsd(self, rv64):
        assert len(rv64.asm("fsd f0, 0(x1)")) == 4

    def test_fadd_s(self, rv64):
        assert len(rv64.asm("fadd.s f0, f1, f2")) == 4

    def test_fsub_s(self, rv64):
        assert len(rv64.asm("fsub.s f0, f1, f2")) == 4

    def test_fmul_s(self, rv64):
        assert len(rv64.asm("fmul.s f0, f1, f2")) == 4

    def test_fdiv_s(self, rv64):
        assert len(rv64.asm("fdiv.s f0, f1, f2")) == 4

    def test_fadd_d(self, rv64):
        assert len(rv64.asm("fadd.d f0, f1, f2")) == 4

    def test_fmadd_s(self, rv64):
        assert len(rv64.asm("fmadd.s f0, f1, f2, f3")) == 4

    def test_fmadd_d(self, rv64):
        assert len(rv64.asm("fmadd.d f0, f1, f2, f3")) == 4

    def test_fcvt_s_w(self, rv64):
        assert len(rv64.asm("fcvt.s.w f0, x1")) == 4

    def test_fcvt_w_s(self, rv64):
        assert len(rv64.asm("fcvt.w.s x1, f0")) == 4

    def test_fcvt_d_s(self, rv64):
        assert len(rv64.asm("fcvt.d.s f0, f1")) == 4

    def test_fmv_x_w(self, rv64):
        assert len(rv64.asm("fmv.x.w x1, f0")) == 4

    def test_feq(self, rv64):
        assert len(rv64.asm("feq.s x1, f0, f1")) == 4

    def test_flt(self, rv64):
        assert len(rv64.asm("flt.d x1, f0, f1")) == 4

    def test_fsqrt(self, rv64):
        assert len(rv64.asm("fsqrt.d f0, f1")) == 4


# ---------------------------------------------------------------------------
# System / CSR
# ---------------------------------------------------------------------------

class TestRV64System:
    def test_ecall(self, rv64):
        assert len(rv64.asm("ecall")) == 4

    def test_ebreak(self, rv64):
        assert len(rv64.asm("ebreak")) == 4

    def test_fence(self, rv64):
        assert len(rv64.asm("fence")) == 4

    def test_fence_i(self, rv64):
        rv = Assembler(
            triple="riscv64",
            cpu="generic-rv64",
            features="+m,+a,+f,+d,+zifencei",
        )
        assert len(rv.asm("fence.i")) == 4

    def test_csrr(self, rv64):
        assert len(rv64.asm("csrr x1, cycle")) == 4

    def test_csrw(self, rv64):
        assert len(rv64.asm("csrw sscratch, x1")) == 4

    def test_csrrw(self, rv64):
        assert len(rv64.asm("csrrw x1, sscratch, x2")) == 4


# ---------------------------------------------------------------------------
# RV64-specific (word-width ops)
# ---------------------------------------------------------------------------

class TestRV64Specific:
    def test_addiw(self, rv64):
        assert len(rv64.asm("addiw x1, x2, 1")) == 4

    def test_addw(self, rv64):
        assert len(rv64.asm("addw x1, x2, x3")) == 4

    def test_subw(self, rv64):
        assert len(rv64.asm("subw x1, x2, x3")) == 4

    def test_sllw(self, rv64):
        assert len(rv64.asm("sllw x1, x2, x3")) == 4

    def test_srlw(self, rv64):
        assert len(rv64.asm("srlw x1, x2, x3")) == 4

    def test_sraw(self, rv64):
        assert len(rv64.asm("sraw x1, x2, x3")) == 4


# ---------------------------------------------------------------------------
# RV32 basics
# ---------------------------------------------------------------------------

class TestRV32:
    def test_nop(self, rv32):
        assert len(rv32.asm("nop")) == 4

    def test_add(self, rv32):
        assert len(rv32.asm("add x1, x2, x3")) == 4

    def test_lw_sw(self, rv32):
        assert len(rv32.asm("lw x1, 0(x2)\nsw x1, 0(x2)")) == 8

    def test_mul(self, rv32):
        assert len(rv32.asm("mul x1, x2, x3")) == 4

    def test_fadd_s(self, rv32):
        assert len(rv32.asm("fadd.s f0, f1, f2")) == 4

    def test_lr_sc_w(self, rv32):
        assert len(rv32.asm("lr.w x1, (x2)")) == 4
        assert len(rv32.asm("sc.w x1, x3, (x2)")) == 4


# ---------------------------------------------------------------------------
# ABI register names
# ---------------------------------------------------------------------------

class TestABINames:
    def test_abi_names(self, rv64):
        """RISC-V ABI names (a0-a7, t0-t6, s0-s11, etc.)."""
        assert len(rv64.asm("add a0, a1, a2")) == 4
        assert len(rv64.asm("add t0, t1, t2")) == 4
        assert len(rv64.asm("add s0, s1, s2")) == 4
        assert len(rv64.asm("mv ra, sp")) == 4
        assert len(rv64.asm("mv gp, tp")) == 4

    def test_fp_abi_names(self, rv64):
        """FP ABI names (fa0-fa7, ft0-ft11, fs0-fs11)."""
        assert len(rv64.asm("fadd.d fa0, fa1, fa2")) == 4
        assert len(rv64.asm("fadd.d ft0, ft1, ft2")) == 4
        assert len(rv64.asm("fadd.d fs0, fs1, fs2")) == 4


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_forward_label(self, rv64):
        code = rv64.asm("jal x1, target\ntarget:\nnop")
        assert len(code) == 8

    def test_backward_label(self, rv64):
        code = rv64.asm("loop:\nnop\njal x0, loop")
        assert len(code) == 8

    def test_local_labels(self, rv64):
        code = rv64.asm("1:\nnop\nbne x1, x0, 1b")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_instruction(self, rv64):
        with pytest.raises(AsmError):
            rv64.asm("not_real x0, x1")

    def test_bad_register(self, rv64):
        with pytest.raises(AsmError):
            rv64.asm("add x99, x0, x1")

    def test_rv64_on_rv32(self, rv32):
        """ld/sd are RV64 only — should fail on RV32."""
        with pytest.raises(AsmError):
            rv32.asm("ld x1, 0(x2)")
