"""Tests for RISC-V assembly (RV32 and RV64).

Comprehensive corpus covering base integer, M/A/F/D extensions,
CSR ops, pseudoinstructions, and all register variants.
"""

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


# ===================================================================
# Helpers
# ===================================================================

# Representative x-register triples
REG_TRIPLES = [
    ("x1", "x2", "x3"),
    ("x0", "x1", "x2"),
    ("x10", "x11", "x12"),
    ("x20", "x21", "x22"),
    ("x28", "x29", "x30"),
    ("x31", "x0", "x1"),
    ("x15", "x16", "x17"),
    ("x5", "x6", "x7"),
]

# ABI name triples
ABI_TRIPLES = [
    ("a0", "a1", "a2"),
    ("a3", "a4", "a5"),
    ("a6", "a7", "t0"),
    ("t1", "t2", "t3"),
    ("t4", "t5", "t6"),
    ("s0", "s1", "s2"),
    ("s3", "s4", "s5"),
    ("s6", "s7", "s8"),
    ("s9", "s10", "s11"),
    ("ra", "sp", "gp"),
    ("tp", "t0", "t1"),
    ("fp", "a0", "a1"),
]

REG_PAIRS = [
    ("x1", "x2"),
    ("x10", "x11"),
    ("x20", "x21"),
    ("x28", "x29"),
    ("x31", "x0"),
    ("a0", "a1"),
    ("t0", "t1"),
    ("s0", "s1"),
    ("ra", "sp"),
]

# FP register triples
FP_TRIPLES = [
    ("f0", "f1", "f2"),
    ("f10", "f11", "f12"),
    ("f20", "f21", "f22"),
    ("f30", "f31", "f0"),
    ("fa0", "fa1", "fa2"),
    ("fa3", "fa4", "fa5"),
    ("ft0", "ft1", "ft2"),
    ("ft6", "ft7", "ft8"),
    ("fs0", "fs1", "fs2"),
    ("fs8", "fs9", "fs10"),
]

FP_PAIRS = [
    ("f0", "f1"),
    ("f10", "f11"),
    ("f20", "f21"),
    ("fa0", "fa1"),
    ("ft0", "ft1"),
    ("fs0", "fs1"),
]

IMMEDIATES_12BIT = [0, 1, -1, 42, 0x7FF, -2048, 255, -128, 100, 2047]
LOAD_OFFSETS = [0, 4, -4, 8, -8, 127, -128, 2047, -2048, 256]


# ===================================================================
# RV64I — ALU register-register (all ops x register combos)
# ===================================================================

RV64_ALU_OPS = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"]
RV64_ALU_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in RV64_ALU_OPS
    for rd, rs1, rs2 in REG_TRIPLES
]


@pytest.mark.parametrize("instr,desc", RV64_ALU_TESTS, ids=[d for _, d in RV64_ALU_TESTS])
def test_rv64_alu_reg(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ABI name variants
RV64_ALU_ABI_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_abi_{rd}_{rs1}_{rs2}")
    for op in ["add", "sub", "and", "or", "xor"]
    for rd, rs1, rs2 in ABI_TRIPLES
]


@pytest.mark.parametrize("instr,desc", RV64_ALU_ABI_TESTS, ids=[d for _, d in RV64_ALU_ABI_TESTS])
def test_rv64_alu_abi(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64I — ALU immediate
# ===================================================================

RV64_IMM_OPS = ["addi", "andi", "ori", "xori", "slti", "sltiu"]
RV64_IMM_TESTS = [
    (f"{op} {rd}, {rs1}, {imm}", f"{op}_{rd}_{rs1}_{imm}")
    for op in RV64_IMM_OPS
    for rd, rs1 in REG_PAIRS
    for imm in IMMEDIATES_12BIT
]


@pytest.mark.parametrize("instr,desc", RV64_IMM_TESTS, ids=[d for _, d in RV64_IMM_TESTS])
def test_rv64_alu_imm(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64I — Shift immediate
# ===================================================================

SHIFT_IMM_OPS = ["slli", "srli", "srai"]
SHIFT_AMOUNTS_64 = [0, 1, 7, 16, 31, 63]
SHIFT_IMM_TESTS = [
    (f"{op} {rd}, {rs1}, {amt}", f"{op}_{rd}_{rs1}_{amt}")
    for op in SHIFT_IMM_OPS
    for rd, rs1 in [("x1", "x2"), ("x10", "x11"), ("a0", "a1"), ("t0", "t1")]
    for amt in SHIFT_AMOUNTS_64
]


@pytest.mark.parametrize("instr,desc", SHIFT_IMM_TESTS, ids=[d for _, d in SHIFT_IMM_TESTS])
def test_rv64_shift_imm(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64I — Upper immediate
# ===================================================================

UPPER_IMM_TESTS = [
    (f"{op} {rd}, {imm}", f"{op}_{rd}_{imm:#x}")
    for op in ["lui", "auipc"]
    for rd in ["x1", "x10", "x31", "a0", "t0", "s0", "ra"]
    for imm in [0, 1, 0x12345, 0xFFFFF, 0x80000]
]


@pytest.mark.parametrize("instr,desc", UPPER_IMM_TESTS, ids=[d for _, d in UPPER_IMM_TESTS])
def test_rv64_upper_imm(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64-specific word-width ops
# ===================================================================

RV64W_REG_OPS = ["addw", "subw", "sllw", "srlw", "sraw"]
RV64W_REG_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in RV64W_REG_OPS
    for rd, rs1, rs2 in REG_TRIPLES[:5]
]

RV64W_IMM_OPS = ["addiw"]
RV64W_IMM_TESTS = [
    (f"{op} {rd}, {rs1}, {imm}", f"{op}_{rd}_{rs1}_{imm}")
    for op in RV64W_IMM_OPS
    for rd, rs1 in REG_PAIRS[:5]
    for imm in IMMEDIATES_12BIT
]

RV64W_SHIFT_OPS = ["slliw", "srliw", "sraiw"]
RV64W_SHIFT_TESTS = [
    (f"{op} {rd}, {rs1}, {amt}", f"{op}_{rd}_{rs1}_{amt}")
    for op in RV64W_SHIFT_OPS
    for rd, rs1 in [("x1", "x2"), ("x10", "x11"), ("a0", "a1")]
    for amt in [0, 1, 7, 15, 31]
]

RV64W_ALL = RV64W_REG_TESTS + RV64W_IMM_TESTS + RV64W_SHIFT_TESTS


@pytest.mark.parametrize("instr,desc", RV64W_ALL, ids=[d for _, d in RV64W_ALL])
def test_rv64w(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64I — Loads and stores (all sizes x offsets)
# ===================================================================

LOAD_OPS = ["lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"]
STORE_OPS = ["sb", "sh", "sw", "sd"]

LOAD_TESTS = [
    (f"{op} {rd}, {off}({rs})", f"{op}_{rd}_{off}_{rs}")
    for op in LOAD_OPS
    for rd, rs in [("x1", "x2"), ("x10", "x11"), ("a0", "sp"), ("t0", "s0")]
    for off in LOAD_OFFSETS
]

STORE_TESTS = [
    (f"{op} {rs2}, {off}({rs1})", f"{op}_{rs2}_{off}_{rs1}")
    for op in STORE_OPS
    for rs2, rs1 in [("x1", "x2"), ("x10", "x11"), ("a0", "sp"), ("t0", "s0")]
    for off in LOAD_OFFSETS
]

LOADSTORE_ALL = LOAD_TESTS + STORE_TESTS


@pytest.mark.parametrize("instr,desc", LOADSTORE_ALL, ids=[d for _, d in LOADSTORE_ALL])
def test_rv64_loadstore(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# RV64I — Branches (all conditions x register combos)
# ===================================================================

BRANCH_OPS = ["beq", "bne", "blt", "bge", "bltu", "bgeu"]
BRANCH_TESTS = [
    (f"1:\nnop\n{op} {rs1}, {rs2}, 1b", f"{op}_{rs1}_{rs2}")
    for op in BRANCH_OPS
    for rs1, rs2 in [("x1", "x2"), ("x0", "x1"), ("x10", "x11"),
                     ("a0", "a1"), ("t0", "t1"), ("s0", "s1"),
                     ("x31", "x0"), ("ra", "sp")]
]


@pytest.mark.parametrize("instr,desc", BRANCH_TESTS, ids=[d for _, d in BRANCH_TESTS])
def test_rv64_branches(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 8  # nop + branch


# ===================================================================
# RV64I — JAL / JALR
# ===================================================================

JAL_TESTS = [
    (f"jal {rd}, target\ntarget:\nnop", f"jal_{rd}")
    for rd in ["x1", "x0", "x5", "ra", "t0"]
]

JALR_TESTS = [
    (f"jalr {rd}, {rs}, {off}", f"jalr_{rd}_{rs}_{off}")
    for rd, rs in [("x1", "x2"), ("x0", "x1"), ("ra", "t0"), ("x0", "ra")]
    for off in [0, 4, -4, 100]
]

JAL_ALL = JAL_TESTS + JALR_TESTS


@pytest.mark.parametrize("instr,desc", JAL_ALL, ids=[d for _, d in JAL_ALL])
def test_rv64_jal(rv64, instr, desc):
    assert len(rv64.asm(instr)) > 0


# ===================================================================
# M extension — multiply/divide (all ops x register combos)
# ===================================================================

M_REG_OPS = ["mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"]
M_REG_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in M_REG_OPS
    for rd, rs1, rs2 in REG_TRIPLES[:5]
]

M_WORD_OPS = ["mulw", "divw", "divuw", "remw", "remuw"]
M_WORD_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in M_WORD_OPS
    for rd, rs1, rs2 in REG_TRIPLES[:5]
]

M_ABI_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_abi_{rd}_{rs1}_{rs2}")
    for op in ["mul", "div", "rem"]
    for rd, rs1, rs2 in ABI_TRIPLES[:5]
]

M_ALL = M_REG_TESTS + M_WORD_TESTS + M_ABI_TESTS


@pytest.mark.parametrize("instr,desc", M_ALL, ids=[d for _, d in M_ALL])
def test_rv64_m_ext(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# A extension — atomics (comprehensive)
# ===================================================================

AMO_OPS = ["amoswap", "amoadd", "amoxor", "amoand", "amoor",
           "amomin", "amomax", "amominu", "amomaxu"]

# .d variants
AMO_D_TESTS = [
    (f"{op}.d {rd}, {rs2}, ({rs1})", f"{op}_d_{rd}_{rs2}_{rs1}")
    for op in AMO_OPS
    for rd, rs2, rs1 in [("x1", "x2", "x3"), ("x10", "x11", "x12"),
                         ("a0", "a1", "a2"), ("t0", "t1", "t2")]
]

# .w variants
AMO_W_TESTS = [
    (f"{op}.w {rd}, {rs2}, ({rs1})", f"{op}_w_{rd}_{rs2}_{rs1}")
    for op in AMO_OPS
    for rd, rs2, rs1 in [("x1", "x2", "x3"), ("x10", "x11", "x12"),
                         ("a0", "a1", "a2")]
]

# lr/sc
LRSC_TESTS = [
    (f"lr.d {rd}, ({rs})", f"lr_d_{rd}_{rs}")
    for rd, rs in [("x1", "x2"), ("x10", "x11"), ("a0", "a1")]
] + [
    (f"sc.d {rd}, {rs2}, ({rs1})", f"sc_d_{rd}_{rs2}_{rs1}")
    for rd, rs2, rs1 in [("x1", "x2", "x3"), ("x10", "x11", "x12")]
] + [
    (f"lr.w {rd}, ({rs})", f"lr_w_{rd}_{rs}")
    for rd, rs in [("x1", "x2"), ("x10", "x11"), ("a0", "a1")]
] + [
    (f"sc.w {rd}, {rs2}, ({rs1})", f"sc_w_{rd}_{rs2}_{rs1}")
    for rd, rs2, rs1 in [("x1", "x2", "x3"), ("x10", "x11", "x12")]
]

# Acquire/release variants
AMO_AQ_RL_TESTS = [
    (f"{op}.d.aq {rd}, {rs2}, ({rs1})", f"{op}_d_aq_{rd}")
    for op in ["amoswap", "amoadd"]
    for rd, rs2, rs1 in [("x1", "x2", "x3")]
] + [
    (f"{op}.d.rl {rd}, {rs2}, ({rs1})", f"{op}_d_rl_{rd}")
    for op in ["amoswap", "amoadd"]
    for rd, rs2, rs1 in [("x1", "x2", "x3")]
] + [
    (f"{op}.d.aqrl {rd}, {rs2}, ({rs1})", f"{op}_d_aqrl_{rd}")
    for op in ["amoswap", "amoadd"]
    for rd, rs2, rs1 in [("x1", "x2", "x3")]
] + [
    ("lr.d.aq x1, (x2)", "lr_d_aq"),
    ("lr.d.aqrl x1, (x2)", "lr_d_aqrl"),
    ("sc.d.rl x1, x2, (x3)", "sc_d_rl"),
    ("sc.d.aqrl x1, x2, (x3)", "sc_d_aqrl"),
]

A_ALL = AMO_D_TESTS + AMO_W_TESTS + LRSC_TESTS + AMO_AQ_RL_TESTS


@pytest.mark.parametrize("instr,desc", A_ALL, ids=[d for _, d in A_ALL])
def test_rv64_a_ext(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# F/D extensions — FP load/store
# ===================================================================

FP_LOAD_STORE_TESTS = [
    (f"{op} {fr}, {off}({xr})", f"{op}_{fr}_{off}_{xr}")
    for op in ["flw", "fsw"]
    for fr, xr in [("f0", "x1"), ("f10", "x11"), ("fa0", "sp"), ("ft0", "s0")]
    for off in [0, 4, -4, 128, -128, 2047, -2048]
] + [
    (f"{op} {fr}, {off}({xr})", f"{op}_{fr}_{off}_{xr}")
    for op in ["fld", "fsd"]
    for fr, xr in [("f0", "x1"), ("f10", "x11"), ("fa0", "sp"), ("ft0", "s0")]
    for off in [0, 8, -8, 128, -128, 2047, -2048]
]


@pytest.mark.parametrize("instr,desc", FP_LOAD_STORE_TESTS, ids=[d for _, d in FP_LOAD_STORE_TESTS])
def test_rv64_fp_loadstore(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# F extension — single-precision arithmetic
# ===================================================================

FP_S_ARITH_OPS = ["fadd.s", "fsub.s", "fmul.s", "fdiv.s"]
FP_S_ARITH_TESTS = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in FP_S_ARITH_OPS
    for fd, fs1, fs2 in FP_TRIPLES
]

FP_S_UNARY_OPS = ["fsqrt.s", "fneg.s", "fabs.s"]
FP_S_UNARY_TESTS = [
    (f"{op} {fd}, {fs}", f"{op}_{fd}_{fs}")
    for op in FP_S_UNARY_OPS
    for fd, fs in FP_PAIRS
]

FP_S_MINMAX = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in ["fmin.s", "fmax.s"]
    for fd, fs1, fs2 in FP_TRIPLES[:5]
]

FP_S_SGNJ = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in ["fsgnj.s", "fsgnjn.s", "fsgnjx.s"]
    for fd, fs1, fs2 in FP_TRIPLES[:3]
]

FP_S_MADD = [
    (f"{op} {fd}, {fs1}, {fs2}, {fs3}", f"{op}_{fd}_{fs1}_{fs2}_{fs3}")
    for op in ["fmadd.s", "fmsub.s", "fnmadd.s", "fnmsub.s"]
    for fd, fs1, fs2 in FP_TRIPLES[:3]
    for fs3 in ["f3", "f13", "fa3"]
]

FP_S_ALL = FP_S_ARITH_TESTS + FP_S_UNARY_TESTS + FP_S_MINMAX + FP_S_SGNJ + FP_S_MADD


@pytest.mark.parametrize("instr,desc", FP_S_ALL, ids=[d for _, d in FP_S_ALL])
def test_rv64_fp_single(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# D extension — double-precision arithmetic
# ===================================================================

FP_D_ARITH_OPS = ["fadd.d", "fsub.d", "fmul.d", "fdiv.d"]
FP_D_ARITH_TESTS = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in FP_D_ARITH_OPS
    for fd, fs1, fs2 in FP_TRIPLES
]

FP_D_UNARY_OPS = ["fsqrt.d", "fneg.d", "fabs.d"]
FP_D_UNARY_TESTS = [
    (f"{op} {fd}, {fs}", f"{op}_{fd}_{fs}")
    for op in FP_D_UNARY_OPS
    for fd, fs in FP_PAIRS
]

FP_D_MINMAX = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in ["fmin.d", "fmax.d"]
    for fd, fs1, fs2 in FP_TRIPLES[:5]
]

FP_D_SGNJ = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in ["fsgnj.d", "fsgnjn.d", "fsgnjx.d"]
    for fd, fs1, fs2 in FP_TRIPLES[:3]
]

FP_D_MADD = [
    (f"{op} {fd}, {fs1}, {fs2}, {fs3}", f"{op}_{fd}_{fs1}_{fs2}_{fs3}")
    for op in ["fmadd.d", "fmsub.d", "fnmadd.d", "fnmsub.d"]
    for fd, fs1, fs2 in FP_TRIPLES[:3]
    for fs3 in ["f3", "f13", "fa3"]
]

FP_D_ALL = FP_D_ARITH_TESTS + FP_D_UNARY_TESTS + FP_D_MINMAX + FP_D_SGNJ + FP_D_MADD


@pytest.mark.parametrize("instr,desc", FP_D_ALL, ids=[d for _, d in FP_D_ALL])
def test_rv64_fp_double(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# F/D — comparison instructions
# ===================================================================

FP_CMP_TESTS = [
    (f"{op} {xd}, {fs1}, {fs2}", f"{op}_{xd}_{fs1}_{fs2}")
    for op in ["feq.s", "flt.s", "fle.s", "feq.d", "flt.d", "fle.d"]
    for xd in ["x1", "x10", "a0"]
    for fs1, fs2 in [("f0", "f1"), ("f10", "f11"), ("fa0", "fa1")]
]


@pytest.mark.parametrize("instr,desc", FP_CMP_TESTS, ids=[d for _, d in FP_CMP_TESTS])
def test_rv64_fp_cmp(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# F/D — conversion instructions
# ===================================================================

FP_CVT_TESTS = [
    # int -> float
    ("fcvt.s.w f0, x1", "fcvt_s_w"),
    ("fcvt.s.wu f0, x1", "fcvt_s_wu"),
    ("fcvt.s.l f0, x1", "fcvt_s_l"),
    ("fcvt.s.lu f0, x1", "fcvt_s_lu"),
    ("fcvt.d.w f0, x1", "fcvt_d_w"),
    ("fcvt.d.wu f0, x1", "fcvt_d_wu"),
    ("fcvt.d.l f0, x1", "fcvt_d_l"),
    ("fcvt.d.lu f0, x1", "fcvt_d_lu"),
    # float -> int
    ("fcvt.w.s x1, f0", "fcvt_w_s"),
    ("fcvt.wu.s x1, f0", "fcvt_wu_s"),
    ("fcvt.l.s x1, f0", "fcvt_l_s"),
    ("fcvt.lu.s x1, f0", "fcvt_lu_s"),
    ("fcvt.w.d x1, f0", "fcvt_w_d"),
    ("fcvt.wu.d x1, f0", "fcvt_wu_d"),
    ("fcvt.l.d x1, f0", "fcvt_l_d"),
    ("fcvt.lu.d x1, f0", "fcvt_lu_d"),
    # float <-> float
    ("fcvt.d.s f0, f1", "fcvt_d_s"),
    ("fcvt.s.d f0, f1", "fcvt_s_d"),
    # With ABI names
    ("fcvt.s.w fa0, a0", "fcvt_s_w_abi"),
    ("fcvt.w.s a0, fa0", "fcvt_w_s_abi"),
    ("fcvt.d.w fa0, a0", "fcvt_d_w_abi"),
    ("fcvt.w.d a0, fa0", "fcvt_w_d_abi"),
    # With different registers
    ("fcvt.s.w f10, x10", "fcvt_s_w_r10"),
    ("fcvt.w.s x10, f10", "fcvt_w_s_r10"),
    ("fcvt.d.l f20, x20", "fcvt_d_l_r20"),
    ("fcvt.l.d x20, f20", "fcvt_l_d_r20"),
]


@pytest.mark.parametrize("instr,desc", FP_CVT_TESTS, ids=[d for _, d in FP_CVT_TESTS])
def test_rv64_fp_cvt(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# F/D — move instructions
# ===================================================================

FP_MV_TESTS = [
    ("fmv.x.w x1, f0", "fmv_x_w_x1_f0"),
    ("fmv.w.x f0, x1", "fmv_w_x_f0_x1"),
    ("fmv.x.d x1, f0", "fmv_x_d_x1_f0"),
    ("fmv.d.x f0, x1", "fmv_d_x_f0_x1"),
    ("fmv.x.w a0, fa0", "fmv_x_w_abi"),
    ("fmv.w.x fa0, a0", "fmv_w_x_abi"),
    ("fmv.x.d a0, fa0", "fmv_x_d_abi"),
    ("fmv.d.x fa0, a0", "fmv_d_x_abi"),
    ("fmv.x.w x10, f10", "fmv_x_w_r10"),
    ("fmv.w.x f10, x10", "fmv_w_x_r10"),
    # fclass
    ("fclass.s x1, f0", "fclass_s"),
    ("fclass.d x1, f0", "fclass_d"),
    ("fclass.s a0, fa0", "fclass_s_abi"),
    ("fclass.d a0, fa0", "fclass_d_abi"),
]


@pytest.mark.parametrize("instr,desc", FP_MV_TESTS, ids=[d for _, d in FP_MV_TESTS])
def test_rv64_fp_mv(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# System / CSR
# ===================================================================

CSR_TESTS = [
    # csrr pseudo (csrrs rd, csr, x0)
    ("csrr x1, cycle", "csrr_cycle"),
    ("csrr x1, time", "csrr_time"),
    ("csrr x1, instret", "csrr_instret"),
    ("csrr x1, cycleh", "csrr_cycleh"),
    # csrw pseudo (csrrw x0, csr, rs)
    ("csrw sscratch, x1", "csrw_sscratch"),
    ("csrw sepc, x1", "csrw_sepc"),
    ("csrw scause, x1", "csrw_scause"),
    ("csrw stval, x1", "csrw_stval"),
    # Full CSR ops
    ("csrrw x1, sscratch, x2", "csrrw"),
    ("csrrs x1, sscratch, x2", "csrrs"),
    ("csrrc x1, sscratch, x2", "csrrc"),
    # CSR immediate
    ("csrrwi x1, sscratch, 5", "csrrwi"),
    ("csrrsi x1, sscratch, 3", "csrrsi"),
    ("csrrci x1, sscratch, 1", "csrrci"),
    # ABI names
    ("csrr a0, cycle", "csrr_cycle_abi"),
    ("csrw sscratch, a0", "csrw_sscratch_abi"),
]

SYSTEM_TESTS = [
    ("ecall", "ecall"),
    ("ebreak", "ebreak"),
    ("fence", "fence"),
    ("fence rw, rw", "fence_rw_rw"),
    ("fence iorw, iorw", "fence_iorw_iorw"),
    ("fence r, w", "fence_r_w"),
    ("fence w, r", "fence_w_r"),
    ("fence.tso", "fence_tso"),
]

SYS_ALL = CSR_TESTS + SYSTEM_TESTS


@pytest.mark.parametrize("instr,desc", SYS_ALL, ids=[d for _, d in SYS_ALL])
def test_rv64_system(rv64, instr, desc):
    assert len(rv64.asm(instr)) == 4


# ===================================================================
# Pseudoinstructions
# ===================================================================

PSEUDO_TESTS = [
    # mv
    ("mv x1, x2", "mv_x1_x2"),
    ("mv a0, a1", "mv_a0_a1"),
    ("mv t0, t1", "mv_t0_t1"),
    ("mv s0, s1", "mv_s0_s1"),
    # not / neg / seqz / snez / sltz / sgtz
    ("not x1, x2", "not_x1_x2"),
    ("not a0, a1", "not_a0_a1"),
    ("neg x1, x2", "neg_x1_x2"),
    ("neg a0, a1", "neg_a0_a1"),
    ("negw x1, x2", "negw"),
    ("seqz x1, x2", "seqz"),
    ("seqz a0, a1", "seqz_abi"),
    ("snez x1, x2", "snez"),
    ("snez a0, a1", "snez_abi"),
    ("sltz x1, x2", "sltz"),
    ("sgtz x1, x2", "sgtz"),
    ("sext.w x1, x2", "sext_w"),
    ("sext.w a0, a1", "sext_w_abi"),
    # nop
    ("nop", "nop"),
    # ret
    ("ret", "ret"),
    # j pseudo (jal x0, offset)
    ("j target\ntarget:\nnop", "j_pseudo"),
    # jr pseudo (jalr x0, rs, 0)
    ("jr x1", "jr_x1"),
    ("jr ra", "jr_ra"),
    ("jr t0", "jr_t0"),
    # call pseudo
    ("call target\ntarget:\nnop", "call_pseudo"),
    # tail pseudo
    ("tail target\ntarget:\nnop", "tail_pseudo"),
]

# li with various immediates
LI_TESTS = [
    (f"li {rd}, {imm}", f"li_{rd}_{imm}")
    for rd in ["x1", "a0", "t0", "s0"]
    for imm in [0, 1, -1, 42, 255, -128, 0x7FF, -2048, 0x1000,
                0xFFFFF, -0x80000, 0x7FFFFFFF, -1]
]

PSEUDO_ALL = PSEUDO_TESTS + LI_TESTS


@pytest.mark.parametrize("instr,desc", PSEUDO_ALL, ids=[d for _, d in PSEUDO_ALL])
def test_rv64_pseudo(rv64, instr, desc):
    assert len(rv64.asm(instr)) > 0


# Branch pseudos with labels
BRANCH_PSEUDO_TESTS = [
    (f"{op} {rs}, target\ntarget:\nnop", f"{op}_{rs}")
    for op in ["beqz", "bnez"]
    for rs in ["x1", "a0", "t0", "s0"]
] + [
    (f"{op} {rs}, target\ntarget:\nnop", f"{op}_{rs}")
    for op in ["blez", "bgez", "bltz", "bgtz"]
    for rs in ["x1", "a0", "t0"]
]


@pytest.mark.parametrize("instr,desc", BRANCH_PSEUDO_TESTS, ids=[d for _, d in BRANCH_PSEUDO_TESTS])
def test_rv64_branch_pseudo(rv64, instr, desc):
    assert len(rv64.asm(instr)) > 0


# ===================================================================
# Labels — local numeric
# ===================================================================

LABEL_TESTS = [
    ("1:\nnop\nbne x1, x0, 1b", "numeric_backward"),
    ("beq x1, x0, 1f\nnop\n1:\nnop", "numeric_forward"),
    ("1:\nnop\n2:\nnop\nbne x1, x0, 1b", "multi_numeric"),
    ("loop:\nnop\nbne x1, x0, loop", "named_backward"),
    ("jal x1, target\ntarget:\nnop", "named_forward"),
    ("start:\nnop\nj start", "j_named_backward"),
]


@pytest.mark.parametrize("instr,desc", LABEL_TESTS, ids=[d for _, d in LABEL_TESTS])
def test_rv64_labels(rv64, instr, desc):
    assert len(rv64.asm(instr)) > 0


# ===================================================================
# RV32 — comprehensive
# ===================================================================

RV32_TESTS = [
    ("nop", "nop"),
    ("add x1, x2, x3", "add"),
    ("sub x1, x2, x3", "sub"),
    ("and x1, x2, x3", "and"),
    ("or x1, x2, x3", "or"),
    ("xor x1, x2, x3", "xor"),
    ("sll x1, x2, x3", "sll"),
    ("srl x1, x2, x3", "srl"),
    ("sra x1, x2, x3", "sra"),
    ("slt x1, x2, x3", "slt"),
    ("sltu x1, x2, x3", "sltu"),
    ("addi x1, x2, 42", "addi"),
    ("andi x1, x2, 0xff", "andi"),
    ("ori x1, x2, 0xff", "ori"),
    ("xori x1, x2, 0xff", "xori"),
    ("slti x1, x2, 42", "slti"),
    ("sltiu x1, x2, 42", "sltiu"),
    ("slli x1, x2, 4", "slli"),
    ("srli x1, x2, 4", "srli"),
    ("srai x1, x2, 4", "srai"),
    ("lui x1, 0x12345", "lui"),
    ("auipc x1, 0", "auipc"),
    ("lw x1, 0(x2)", "lw"),
    ("sw x1, 0(x2)", "sw"),
    ("lb x1, 0(x2)", "lb"),
    ("lbu x1, 0(x2)", "lbu"),
    ("lh x1, 0(x2)", "lh"),
    ("lhu x1, 0(x2)", "lhu"),
    ("sb x1, 0(x2)", "sb"),
    ("sh x1, 0(x2)", "sh"),
    # M extension
    ("mul x1, x2, x3", "mul"),
    ("mulh x1, x2, x3", "mulh"),
    ("div x1, x2, x3", "div"),
    ("divu x1, x2, x3", "divu"),
    ("rem x1, x2, x3", "rem"),
    ("remu x1, x2, x3", "remu"),
    # A extension
    ("lr.w x1, (x2)", "lr_w"),
    ("sc.w x1, x3, (x2)", "sc_w"),
    ("amoswap.w x1, x2, (x3)", "amoswap_w"),
    ("amoadd.w x1, x2, (x3)", "amoadd_w"),
    # F extension
    ("flw f0, 0(x1)", "flw"),
    ("fsw f0, 0(x1)", "fsw"),
    ("fadd.s f0, f1, f2", "fadd_s"),
    ("fsub.s f0, f1, f2", "fsub_s"),
    ("fmul.s f0, f1, f2", "fmul_s"),
    ("fdiv.s f0, f1, f2", "fdiv_s"),
    ("fsqrt.s f0, f1", "fsqrt_s"),
    ("fcvt.s.w f0, x1", "fcvt_s_w"),
    ("fcvt.w.s x1, f0", "fcvt_w_s"),
    ("feq.s x1, f0, f1", "feq_s"),
    ("flt.s x1, f0, f1", "flt_s"),
    ("fle.s x1, f0, f1", "fle_s"),
    # Branches
    ("1:\nnop\nbeq x1, x2, 1b", "beq"),
    ("1:\nnop\nbne x1, x2, 1b", "bne"),
    ("1:\nnop\nblt x1, x2, 1b", "blt"),
    ("1:\nnop\nbge x1, x2, 1b", "bge"),
    # System
    ("ecall", "ecall"),
    ("ebreak", "ebreak"),
    ("fence", "fence"),
    # Pseudos
    ("nop", "nop_p"),
    ("ret", "ret_p"),
    ("mv x1, x2", "mv"),
    ("li x1, 42", "li"),
    ("not x1, x2", "not"),
    ("neg x1, x2", "neg"),
    # ABI names
    ("add a0, a1, a2", "add_abi"),
    ("lw s0, 0(sp)", "lw_abi"),
    ("sw ra, 0(sp)", "sw_abi"),
]


@pytest.mark.parametrize("instr,desc", RV32_TESTS, ids=[d for _, d in RV32_TESTS])
def test_rv32(rv32, instr, desc):
    assert len(rv32.asm(instr)) > 0


# ===================================================================
# RV32 — error cases
# ===================================================================

RV32_ERROR_TESTS = [
    ("ld x1, 0(x2)", "ld_on_rv32"),
    ("sd x1, 0(x2)", "sd_on_rv32"),
    ("lwu x1, 0(x2)", "lwu_on_rv32"),
    ("addw x1, x2, x3", "addw_on_rv32"),
    ("subw x1, x2, x3", "subw_on_rv32"),
    ("addiw x1, x2, 1", "addiw_on_rv32"),
]


@pytest.mark.parametrize("instr,desc", RV32_ERROR_TESTS, ids=[d for _, d in RV32_ERROR_TESTS])
def test_rv32_rejects_rv64(rv32, instr, desc):
    with pytest.raises(AsmError):
        rv32.asm(instr)


# ===================================================================
# Error cases (RV64)
# ===================================================================

ERROR_TESTS = [
    ("not_real x0, x1", "bad_instruction"),
    ("add x99, x0, x1", "bad_register"),
    ("add x1, x2", "missing_operand"),
    ("fadd.q f0, f1, f2", "bad_fp_suffix"),
]


@pytest.mark.parametrize("instr,desc", ERROR_TESTS, ids=[d for _, d in ERROR_TESTS])
def test_rv64_errors(rv64, instr, desc):
    with pytest.raises(AsmError):
        rv64.asm(instr)


# ===================================================================
# Multi-instruction sequences
# ===================================================================

SEQUENCE_TESTS = [
    ("addi sp, sp, -16\nsd ra, 8(sp)\nsd s0, 0(sp)", "prologue"),
    ("ld ra, 8(sp)\nld s0, 0(sp)\naddi sp, sp, 16\nret", "epilogue"),
    ("li a7, 93\nli a0, 0\necall", "exit_syscall"),
    ("add a0, a1, a2\nret", "add_and_return"),
    ("mul a0, a1, a2\nadd a0, a0, a3\nret", "mac_return"),
    ("flw fa0, 0(a0)\nflw fa1, 4(a0)\nfadd.s fa0, fa0, fa1\nfsw fa0, 0(a0)", "fp_add_mem"),
    ("1:\naddi a0, a0, -1\nbnez a0, 1b\nret", "countdown_loop"),
]


@pytest.mark.parametrize("instr,desc", SEQUENCE_TESTS, ids=[d for _, d in SEQUENCE_TESTS])
def test_rv64_sequences(rv64, instr, desc):
    assert len(rv64.asm(instr)) > 0


# ===================================================================
# Zifencei extension
# ===================================================================


class TestZifencei:
    def test_fence_i(self):
        rv = Assembler(
            triple="riscv64",
            cpu="generic-rv64",
            features="+m,+a,+f,+d,+zifencei",
        )
        assert len(rv.asm("fence.i")) == 4
