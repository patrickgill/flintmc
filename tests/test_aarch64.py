"""Tests for AArch64 (ARMv8-A 64-bit) assembly.

Comprehensive corpus covering arithmetic, logic, loads/stores, SIMD/NEON,
conditional ops, atomics, system instructions, and addressing modes.
"""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def asm():
    return Assembler.aarch64()


@pytest.fixture(scope="module")
def asm_lse():
    return Assembler(triple="aarch64", features="+lse")


@pytest.fixture(scope="module")
def asm_fullfp16():
    return Assembler(triple="aarch64", features="+fullfp16")


@pytest.fixture(scope="module")
def asm_crc():
    return Assembler(triple="aarch64", features="+crc")


# ===================================================================
# Helpers
# ===================================================================

# Representative 64-bit register triples
X_TRIPLES = [
    ("x0", "x1", "x2"),
    ("x3", "x4", "x5"),
    ("x10", "x11", "x12"),
    ("x20", "x21", "x22"),
    ("x28", "x29", "x30"),
    ("x15", "x16", "x17"),
]

# Representative 32-bit register triples
W_TRIPLES = [
    ("w0", "w1", "w2"),
    ("w3", "w4", "w5"),
    ("w10", "w11", "w12"),
    ("w20", "w21", "w22"),
    ("w28", "w29", "w30"),
]

X_PAIRS = [
    ("x0", "x1"),
    ("x10", "x11"),
    ("x20", "x21"),
    ("x28", "x29"),
    ("x30", "x0"),
]

W_PAIRS = [
    ("w0", "w1"),
    ("w10", "w11"),
    ("w20", "w21"),
    ("w28", "w29"),
]

# Condition codes
CONDS = ["eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl",
         "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al"]

# SIMD/FP register combos
D_TRIPLES = [("d0", "d1", "d2"), ("d10", "d11", "d12"), ("d20", "d21", "d22")]
S_TRIPLES = [("s0", "s1", "s2"), ("s10", "s11", "s12"), ("s20", "s21", "s22")]


# ===================================================================
# Basic known-byte tests
# ===================================================================

class TestBasicBytes:
    def test_nop(self, asm):
        assert asm.asm("nop") == b"\x1f\x20\x03\xd5"

    def test_ret(self, asm):
        assert asm.asm("ret") == b"\xc0\x03\x5f\xd6"


# ===================================================================
# Arithmetic — register-register (64-bit and 32-bit)
# ===================================================================

ARITH_OPS_3REG = ["add", "sub", "adds", "subs"]
ARITH_X_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_x_{rd}_{rs1}_{rs2}")
    for op in ARITH_OPS_3REG
    for rd, rs1, rs2 in X_TRIPLES
]
ARITH_W_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_w_{rd}_{rs1}_{rs2}")
    for op in ARITH_OPS_3REG
    for rd, rs1, rs2 in W_TRIPLES
]

ARITH_OTHER = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in ["mul", "sdiv", "udiv", "mneg", "smulh", "umulh"]
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("x10", "x11", "x12")]
] + [
    # madd/msub need 4 operands
    (f"madd {rd}, {rs1}, {rs2}, {rs3}", f"madd_{rd}_{rs1}_{rs2}_{rs3}")
    for rd, rs1, rs2 in X_TRIPLES[:3]
    for rs3 in ["x3", "x13"]
] + [
    (f"msub {rd}, {rs1}, {rs2}, {rs3}", f"msub_{rd}_{rs1}_{rs2}_{rs3}")
    for rd, rs1, rs2 in X_TRIPLES[:2]
    for rs3 in ["x3"]
] + [
    # smull/umull use w source registers
    (f"smull x0, w1, w2", "smull_x0_w1_w2"),
    (f"smull x10, w11, w12", "smull_x10_w11_w12"),
    (f"umull x0, w1, w2", "umull_x0_w1_w2"),
    (f"umull x10, w11, w12", "umull_x10_w11_w12"),
]

# neg pseudo
ARITH_NEG = [
    (f"neg {rd}, {rs}", f"neg_{rd}_{rs}")
    for rd, rs in X_PAIRS
] + [
    (f"neg {rd}, {rs}", f"neg_w_{rd}_{rs}")
    for rd, rs in W_PAIRS
]

# Immediate arithmetic
ARITH_IMM_VALS = [0, 1, 42, 255, 0xFFF]
ARITH_IMM_TESTS = [
    (f"{op} {rd}, {rs}, #{imm}", f"{op}_imm_{rd}_{rs}_{imm}")
    for op in ["add", "sub", "adds", "subs"]
    for rd, rs in [("x0", "x1"), ("x10", "x11"), ("w0", "w1"), ("w10", "w11")]
    for imm in ARITH_IMM_VALS
]

# Shifted arithmetic
ARITH_SHIFTED = [
    (f"add x0, x1, x2, lsl #{s}", f"add_lsl_{s}")
    for s in [0, 1, 4, 16, 32, 63]
] + [
    (f"add x0, x1, x2, lsr #{s}", f"add_lsr_{s}")
    for s in [1, 4, 16, 32]
] + [
    (f"add x0, x1, x2, asr #{s}", f"add_asr_{s}")
    for s in [1, 4, 16, 32]
] + [
    (f"sub x0, x1, x2, lsl #{s}", f"sub_lsl_{s}")
    for s in [0, 4, 16]
]

ARITH_ALL = ARITH_X_TESTS + ARITH_W_TESTS + ARITH_OTHER + ARITH_NEG + ARITH_IMM_TESTS + ARITH_SHIFTED


@pytest.mark.parametrize("instr,desc", ARITH_ALL, ids=[d for _, d in ARITH_ALL])
def test_arith(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Logic and bit manipulation
# ===================================================================

LOGIC_OPS_3REG = ["and", "orr", "eor", "orn", "bic", "eon"]
LOGIC_X_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_{rd}_{rs1}_{rs2}")
    for op in LOGIC_OPS_3REG
    for rd, rs1, rs2 in X_TRIPLES[:4]
]
LOGIC_W_TESTS = [
    (f"{op} {rd}, {rs1}, {rs2}", f"{op}_w_{rd}_{rs1}_{rs2}")
    for op in ["and", "orr", "eor", "bic"]
    for rd, rs1, rs2 in W_TRIPLES[:3]
]

LOGIC_UNARY = [
    (f"mvn {rd}, {rs}", f"mvn_{rd}_{rs}")
    for rd, rs in X_PAIRS
] + [
    (f"mvn {rd}, {rs}", f"mvn_w_{rd}_{rs}")
    for rd, rs in W_PAIRS[:3]
]

# Logic immediate
LOGIC_IMM_TESTS = [
    (f"and x0, x1, #{imm}", f"and_imm_{imm:#x}")
    for imm in [0xff, 0xffff, 0xff00ff00ff00ff00, 0x5555555555555555]
] + [
    (f"orr x0, x1, #{imm}", f"orr_imm_{imm:#x}")
    for imm in [0xff, 0xffff, 0xff00ff00ff00ff00]
] + [
    (f"eor x0, x1, #{imm}", f"eor_imm_{imm:#x}")
    for imm in [0xff, 0xffff]
] + [
    (f"tst x0, #{imm}", f"tst_imm_{imm:#x}")
    for imm in [0xff, 0xffff, 0x1, 0x5555555555555555]
] + [
    (f"tst {r1}, {r2}", f"tst_{r1}_{r2}")
    for r1, r2 in [("x0", "x1"), ("x10", "x11"), ("w0", "w1")]
]

# Shifted logic
LOGIC_SHIFTED = [
    (f"and x0, x1, x2, lsl #{s}", f"and_lsl_{s}")
    for s in [0, 4, 16, 32]
] + [
    (f"orr x0, x1, x2, lsr #{s}", f"orr_lsr_{s}")
    for s in [4, 16, 32]
]

LOGIC_ALL = LOGIC_X_TESTS + LOGIC_W_TESTS + LOGIC_UNARY + LOGIC_IMM_TESTS + LOGIC_SHIFTED


@pytest.mark.parametrize("instr,desc", LOGIC_ALL, ids=[d for _, d in LOGIC_ALL])
def test_logic(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Shifts and bit ops
# ===================================================================

SHIFT_TESTS = [
    (f"{op} {rd}, {rs}, #{amt}", f"{op}_{rd}_{rs}_{amt}")
    for op in ["lsl", "lsr", "asr"]
    for rd, rs in [("x0", "x1"), ("x10", "x11"), ("w0", "w1"), ("w10", "w11")]
    for amt in [1, 4, 16, 31]
] + [
    (f"{op} {rd}, {rs}, #{amt}", f"{op}_x64_{rd}_{rs}_{amt}")
    for op in ["lsl", "lsr", "asr"]
    for rd, rs in [("x0", "x1")]
    for amt in [32, 48, 63]
] + [
    (f"ror {rd}, {rs1}, {rs2}", f"ror_reg_{rd}_{rs1}_{rs2}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("w0", "w1", "w2")]
] + [
    (f"ror {rd}, {rs}, #{amt}", f"ror_imm_{rd}_{rs}_{amt}")
    for rd, rs in [("x0", "x1")]
    for amt in [1, 4, 16, 63]
]

BIT_OPS = [
    (f"{op} {rd}, {rs}", f"{op}_{rd}_{rs}")
    for op in ["clz", "cls", "rbit", "rev", "rev16", "rev32"]
    for rd, rs in [("x0", "x1"), ("x10", "x11")]
] + [
    (f"{op} {rd}, {rs}", f"{op}_w_{rd}_{rs}")
    for op in ["clz", "cls", "rbit", "rev", "rev16"]
    for rd, rs in [("w0", "w1")]
]

SHIFT_ALL = SHIFT_TESTS + BIT_OPS


@pytest.mark.parametrize("instr,desc", SHIFT_ALL, ids=[d for _, d in SHIFT_ALL])
def test_shifts(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Load / store — comprehensive addressing modes
# ===================================================================

# Base addressing
LOAD_BASE_TESTS = [
    (f"ldr {rt}, [{rn}]", f"ldr_{rt}_{rn}")
    for rt, rn in [("x0", "x1"), ("x0", "sp"), ("x10", "x28"),
                   ("w0", "x1"), ("w0", "sp")]
]

# Immediate offset
LOAD_IMM_TESTS = [
    (f"ldr x0, [x1, #{off}]", f"ldr_x_off_{off}")
    for off in [0, 8, 16, 64, 256, 32760]
] + [
    (f"ldr w0, [x1, #{off}]", f"ldr_w_off_{off}")
    for off in [0, 4, 8, 64, 16380]
] + [
    (f"str x0, [x1, #{off}]", f"str_x_off_{off}")
    for off in [0, 8, 16, 256]
] + [
    (f"str w0, [x1, #{off}]", f"str_w_off_{off}")
    for off in [0, 4, 8, 64]
]

# Pre-index and post-index
LOAD_PREPOST = [
    (f"ldr x0, [x1, #{off}]!", f"ldr_pre_{off}")
    for off in [8, 16, -16, -256]
] + [
    (f"ldr x0, [x1], #{off}", f"ldr_post_{off}")
    for off in [8, 16, -16, -256]
] + [
    (f"str x0, [x1, #{off}]!", f"str_pre_{off}")
    for off in [8, -16]
] + [
    (f"str x0, [x1], #{off}", f"str_post_{off}")
    for off in [8, -16]
]

# Register offset
LOAD_REG_OFF = [
    (f"ldr x0, [x1, {rm}]", f"ldr_reg_{rm}")
    for rm in ["x2", "x10", "x20"]
] + [
    (f"ldr x0, [x1, x2, lsl #3]", "ldr_reg_lsl3"),
    (f"ldr w0, [x1, x2, lsl #2]", "ldr_w_reg_lsl2"),
    (f"ldr x0, [x1, w2, sxtw]", "ldr_sxtw"),
    (f"ldr x0, [x1, w2, uxtw]", "ldr_uxtw"),
    (f"ldr x0, [x1, w2, sxtw #3]", "ldr_sxtw_lsl3"),
    (f"ldr x0, [x1, w2, uxtw #3]", "ldr_uxtw_lsl3"),
]

# All widths
LOAD_WIDTHS = [
    ("ldrb w0, [x1]", "ldrb"),
    ("ldrb w0, [x1, #1]", "ldrb_off"),
    ("ldrh w0, [x1]", "ldrh"),
    ("ldrh w0, [x1, #2]", "ldrh_off"),
    ("ldrsb x0, [x1]", "ldrsb_x"),
    ("ldrsb w0, [x1]", "ldrsb_w"),
    ("ldrsh x0, [x1]", "ldrsh_x"),
    ("ldrsh w0, [x1]", "ldrsh_w"),
    ("ldrsw x0, [x1]", "ldrsw"),
    ("ldrsw x0, [x1, #4]", "ldrsw_off"),
    ("strb w0, [x1]", "strb"),
    ("strb w0, [x1, #1]", "strb_off"),
    ("strh w0, [x1]", "strh"),
    ("strh w0, [x1, #2]", "strh_off"),
]

# Load/store pair
LDP_STP_TESTS = [
    (f"ldp x0, x1, [sp]", "ldp_sp"),
    (f"ldp x0, x1, [sp, #16]", "ldp_sp_off"),
    (f"ldp x0, x1, [sp, #-16]!", "ldp_sp_pre"),
    (f"ldp x0, x1, [sp], #16", "ldp_sp_post"),
    (f"stp x0, x1, [sp]", "stp_sp"),
    (f"stp x0, x1, [sp, #-16]!", "stp_sp_pre"),
    (f"stp x0, x1, [sp], #16", "stp_sp_post"),
    (f"ldp w0, w1, [x2]", "ldp_w"),
    (f"stp w0, w1, [x2]", "stp_w"),
    (f"ldp x10, x11, [x12, #16]", "ldp_r10"),
    (f"stp x10, x11, [x12, #-16]!", "stp_r10_pre"),
    (f"ldp q0, q1, [x0]", "ldp_q"),
    (f"stp q0, q1, [x0]", "stp_q"),
    (f"ldp d0, d1, [x0]", "ldp_d"),
    (f"stp d0, d1, [x0]", "stp_d"),
    (f"ldp s0, s1, [x0]", "ldp_s"),
    (f"stp s0, s1, [x0]", "stp_s"),
]

LOAD_ALL = (LOAD_BASE_TESTS + LOAD_IMM_TESTS + LOAD_PREPOST +
            LOAD_REG_OFF + LOAD_WIDTHS + LDP_STP_TESTS)


@pytest.mark.parametrize("instr,desc", LOAD_ALL, ids=[d for _, d in LOAD_ALL])
def test_loadstore(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Branches and control flow
# ===================================================================

# Conditional branches (all condition codes)
BCOND_TESTS = [
    (f"cmp x0, #0\nb.{cc} target\nnop\ntarget:\nnop", f"b_{cc}")
    for cc in CONDS
]

# CBZ/CBNZ with x and w
CBZ_TESTS = [
    (f"cbz {r}, skip\nnop\nskip:\nnop", f"cbz_{r}")
    for r in ["x0", "x10", "x30", "w0", "w10", "w30"]
] + [
    (f"cbnz {r}, skip\nnop\nskip:\nnop", f"cbnz_{r}")
    for r in ["x0", "x10", "x30", "w0", "w10", "w30"]
]

# TBZ/TBNZ with various bit numbers
TBZ_TESTS = [
    (f"tbz {r}, #{bit}, skip\nnop\nskip:\nnop", f"tbz_{r}_b{bit}")
    for r in ["x0", "x10", "w0"]
    for bit in [0, 1, 7, 15, 31]
] + [
    (f"tbz x0, #{bit}, skip\nnop\nskip:\nnop", f"tbz_x0_b{bit}")
    for bit in [32, 48, 63]
] + [
    (f"tbnz {r}, #{bit}, skip\nnop\nskip:\nnop", f"tbnz_{r}_b{bit}")
    for r in ["x0", "x10", "w0"]
    for bit in [0, 7, 15, 31]
]

# Direct branches
BRANCH_DIRECT = [
    ("b target\nnop\ntarget:\nnop", "b_fwd"),
    ("bl target\ntarget:\nnop", "bl_fwd"),
    ("loop:\nnop\nb loop", "b_back"),
    ("loop:\nnop\nbl loop", "bl_back"),
    ("br x0", "br_x0"),
    ("br x10", "br_x10"),
    ("br x30", "br_x30"),
    ("blr x0", "blr_x0"),
    ("blr x10", "blr_x10"),
    ("blr x30", "blr_x30"),
    ("ret", "ret"),
    ("ret x0", "ret_x0"),
    ("ret x30", "ret_x30"),
]

BRANCH_ALL = BCOND_TESTS + CBZ_TESTS + TBZ_TESTS + BRANCH_DIRECT


@pytest.mark.parametrize("instr,desc", BRANCH_ALL, ids=[d for _, d in BRANCH_ALL])
def test_branches(asm, instr, desc):
    assert len(asm.asm(instr)) > 0


# ===================================================================
# Conditional select / compare
# ===================================================================

CSEL_TESTS = [
    (f"csel {rd}, {rs1}, {rs2}, {cc}", f"csel_{rd}_{cc}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("x10", "x11", "x12"),
                         ("w0", "w1", "w2")]
    for cc in ["eq", "ne", "lt", "ge", "gt", "le", "hi", "lo"]
]

CSINC_TESTS = [
    (f"csinc {rd}, {rs1}, {rs2}, {cc}", f"csinc_{rd}_{cc}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("w0", "w1", "w2")]
    for cc in ["eq", "ne", "lt", "ge"]
]

CSINV_TESTS = [
    (f"csinv {rd}, {rs1}, {rs2}, {cc}", f"csinv_{rd}_{cc}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("w0", "w1", "w2")]
    for cc in ["eq", "ne", "lt", "ge"]
]

CSNEG_TESTS = [
    (f"csneg {rd}, {rs1}, {rs2}, {cc}", f"csneg_{rd}_{cc}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("w0", "w1", "w2")]
    for cc in ["eq", "ne", "lt", "ge"]
]

CSET_TESTS = [
    (f"cset {rd}, {cc}", f"cset_{rd}_{cc}")
    for rd in ["x0", "x10", "w0", "w10"]
    for cc in ["eq", "ne", "lt", "ge", "gt", "le", "hi", "lo", "cs", "cc"]
]

CSETM_TESTS = [
    (f"csetm {rd}, {cc}", f"csetm_{rd}_{cc}")
    for rd in ["x0", "w0"]
    for cc in ["eq", "ne", "lt", "ge"]
]

CINC_TESTS = [
    (f"cinc {rd}, {rs}, {cc}", f"cinc_{rd}_{rs}_{cc}")
    for rd, rs in [("x0", "x1"), ("w0", "w1")]
    for cc in ["eq", "ne", "lt"]
]

CONDSEL_ALL = (CSEL_TESTS + CSINC_TESTS + CSINV_TESTS + CSNEG_TESTS +
               CSET_TESTS + CSETM_TESTS + CINC_TESTS)


@pytest.mark.parametrize("instr,desc", CONDSEL_ALL, ids=[d for _, d in CONDSEL_ALL])
def test_condsel(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Comparisons
# ===================================================================

CMP_TESTS = [
    (f"cmp {r1}, {r2}", f"cmp_{r1}_{r2}")
    for r1, r2 in [("x0", "x1"), ("x10", "x11"), ("x30", "x0"),
                   ("w0", "w1"), ("w10", "w11")]
] + [
    (f"cmp {r}, #{imm}", f"cmp_{r}_{imm}")
    for r in ["x0", "x10", "w0", "w10"]
    for imm in [0, 1, 42, 255, 0xFFF]
] + [
    (f"cmn {r1}, {r2}", f"cmn_{r1}_{r2}")
    for r1, r2 in [("x0", "x1"), ("x10", "x11"), ("w0", "w1")]
] + [
    (f"cmn {r}, #{imm}", f"cmn_{r}_{imm}")
    for r in ["x0", "w0"]
    for imm in [0, 1, 42]
] + [
    (f"ccmp {r1}, {r2}, #{nzcv}, {cc}", f"ccmp_{r1}_{r2}_{nzcv}_{cc}")
    for r1, r2 in [("x0", "x1"), ("w0", "w1")]
    for nzcv in [0, 4, 8, 15]
    for cc in ["eq", "ne"]
]


@pytest.mark.parametrize("instr,desc", CMP_TESTS, ids=[d for _, d in CMP_TESTS])
def test_comparisons(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Move wide / bitfield
# ===================================================================

MOVWIDE_TESTS = [
    (f"movz {rd}, #{imm}", f"movz_{rd}_{imm:#x}")
    for rd in ["x0", "x10", "w0", "w10"]
    for imm in [0, 1, 0x1234, 0xFFFF]
] + [
    (f"movz x0, #{imm}, lsl #{shift}", f"movz_x0_{imm:#x}_lsl{shift}")
    for imm in [0x1234, 0xFFFF]
    for shift in [0, 16, 32, 48]
] + [
    (f"movk {rd}, #{imm}", f"movk_{rd}_{imm:#x}")
    for rd in ["x0", "x10", "w0"]
    for imm in [0x1234, 0x5678, 0xFFFF]
] + [
    (f"movk x0, #{imm}, lsl #{shift}", f"movk_x0_{imm:#x}_lsl{shift}")
    for imm in [0x5678]
    for shift in [0, 16, 32, 48]
] + [
    (f"movn {rd}, #{imm}", f"movn_{rd}_{imm}")
    for rd in ["x0", "x10", "w0"]
    for imm in [0, 1, 0xFFFF]
] + [
    (f"mov {rd}, #{imm}", f"mov_{rd}_{imm}")
    for rd in ["x0", "x10", "w0", "w10"]
    for imm in [0, 1, 42, 255, 0xFFFF, -1]
]

BITFIELD_TESTS = [
    (f"ubfx {rd}, {rs}, #{lsb}, #{width}", f"ubfx_{rd}_{lsb}_{width}")
    for rd, rs in [("x0", "x1"), ("w0", "w1")]
    for lsb, width in [(0, 8), (0, 16), (8, 8), (16, 16)]
] + [
    (f"sbfx {rd}, {rs}, #{lsb}, #{width}", f"sbfx_{rd}_{lsb}_{width}")
    for rd, rs in [("x0", "x1"), ("w0", "w1")]
    for lsb, width in [(0, 8), (0, 16), (8, 8)]
] + [
    (f"bfi {rd}, {rs}, #{lsb}, #{width}", f"bfi_{rd}_{lsb}_{width}")
    for rd, rs in [("x0", "x1"), ("w0", "w1")]
    for lsb, width in [(0, 8), (8, 4), (16, 16)]
] + [
    (f"bfxil {rd}, {rs}, #{lsb}, #{width}", f"bfxil_{rd}_{lsb}_{width}")
    for rd, rs in [("x0", "x1"), ("w0", "w1")]
    for lsb, width in [(0, 8), (8, 8)]
] + [
    (f"extr {rd}, {rs1}, {rs2}, #{rot}", f"extr_{rd}_{rot}")
    for rd, rs1, rs2 in [("x0", "x1", "x2"), ("w0", "w1", "w2")]
    for rot in [4, 16]
] + [
    # Sign/zero extend aliases
    ("sxtb x0, w1", "sxtb_x"),
    ("sxth x0, w1", "sxth_x"),
    ("sxtw x0, w1", "sxtw_x"),
    ("uxtb w0, w1", "uxtb_w"),
    ("uxth w0, w1", "uxth_w"),
]

MOVBF_ALL = MOVWIDE_TESTS + BITFIELD_TESTS


@pytest.mark.parametrize("instr,desc", MOVBF_ALL, ids=[d for _, d in MOVBF_ALL])
def test_movwide_bitfield(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# SIMD / NEON — scalar FP
# ===================================================================

FP_SCALAR_ARITH = [
    (f"{op} {fd}, {fs1}, {fs2}", f"{op}_{fd}_{fs1}_{fs2}")
    for op in ["fadd", "fsub", "fmul", "fdiv", "fmax", "fmin", "fnmul"]
    for fd, fs1, fs2 in D_TRIPLES + S_TRIPLES
]

FP_SCALAR_UNARY = [
    (f"{op} {fd}, {fs}", f"{op}_{fd}_{fs}")
    for op in ["fabs", "fneg", "fsqrt", "frintn", "frintp", "frintm", "frintz"]
    for fd, fs in [("d0", "d1"), ("d10", "d11"), ("s0", "s1"), ("s10", "s11")]
]

FP_SCALAR_MADD = [
    (f"{op} {fd}, {fs1}, {fs2}, {fs3}", f"{op}_{fd}")
    for op in ["fmadd", "fmsub", "fnmadd", "fnmsub"]
    for fd, fs1, fs2 in [("d0", "d1", "d2"), ("s0", "s1", "s2")]
    for fs3 in ["d3", "s3"][:1]  # match precision
]
# Fix: separate d and s fmadd
FP_SCALAR_MADD = [
    (f"{op} d0, d1, d2, d3", f"{op}_d")
    for op in ["fmadd", "fmsub", "fnmadd", "fnmsub"]
] + [
    (f"{op} s0, s1, s2, s3", f"{op}_s")
    for op in ["fmadd", "fmsub", "fnmadd", "fnmsub"]
]

FP_SCALAR_CMP = [
    (f"fcmp {r1}, {r2}", f"fcmp_{r1}_{r2}")
    for r1, r2 in [("d0", "d1"), ("d10", "d11"), ("s0", "s1"), ("s10", "s11")]
] + [
    (f"fcmp {r}, #0.0", f"fcmp_{r}_zero")
    for r in ["d0", "s0"]
] + [
    (f"fcmpe {r1}, {r2}", f"fcmpe_{r1}_{r2}")
    for r1, r2 in [("d0", "d1"), ("s0", "s1")]
]

FP_SCALAR_CVT = [
    ("fcvtzs x0, d0", "fcvtzs_x_d"),
    ("fcvtzs w0, d0", "fcvtzs_w_d"),
    ("fcvtzs x0, s0", "fcvtzs_x_s"),
    ("fcvtzs w0, s0", "fcvtzs_w_s"),
    ("fcvtzu x0, d0", "fcvtzu_x_d"),
    ("fcvtzu w0, d0", "fcvtzu_w_d"),
    ("scvtf d0, x0", "scvtf_d_x"),
    ("scvtf d0, w0", "scvtf_d_w"),
    ("scvtf s0, x0", "scvtf_s_x"),
    ("scvtf s0, w0", "scvtf_s_w"),
    ("ucvtf d0, x0", "ucvtf_d_x"),
    ("ucvtf d0, w0", "ucvtf_d_w"),
    ("ucvtf s0, x0", "ucvtf_s_x"),
    ("ucvtf s0, w0", "ucvtf_s_w"),
    ("fcvt d0, s0", "fcvt_d_s"),
    ("fcvt s0, d0", "fcvt_s_d"),
]

FP_SCALAR_MOV = [
    ("fmov d0, d1", "fmov_d_d"),
    ("fmov s0, s1", "fmov_s_s"),
    ("fmov d0, x0", "fmov_d_x"),
    ("fmov x0, d0", "fmov_x_d"),
    ("fmov s0, w0", "fmov_s_w"),
    ("fmov w0, s0", "fmov_w_s"),
    ("fmov d0, #1.0", "fmov_d_imm1"),
    ("fmov s0, #1.0", "fmov_s_imm1"),
    ("fmov d0, #0.5", "fmov_d_imm05"),
    ("fmov d0, #-1.0", "fmov_d_imm_neg1"),
]

FP_SCALAR_LDR = [
    ("ldr d0, [x0]", "ldr_d"),
    ("ldr s0, [x0]", "ldr_s"),
    ("str d0, [x0]", "str_d"),
    ("str s0, [x0]", "str_s"),
    ("ldr d0, [x0, #8]", "ldr_d_off"),
    ("ldr s0, [x0, #4]", "ldr_s_off"),
    ("ldr d0, [x0, #8]!", "ldr_d_pre"),
    ("ldr d0, [x0], #8", "ldr_d_post"),
]

FP_SCALAR_ALL = (FP_SCALAR_ARITH + FP_SCALAR_UNARY + FP_SCALAR_MADD +
                 FP_SCALAR_CMP + FP_SCALAR_CVT + FP_SCALAR_MOV + FP_SCALAR_LDR)


@pytest.mark.parametrize("instr,desc", FP_SCALAR_ALL, ids=[d for _, d in FP_SCALAR_ALL])
def test_fp_scalar(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# SIMD / NEON — vector operations
# ===================================================================

SIMD_VEC_ARITH = [
    (f"{op} v0.{arr}, v1.{arr}, v2.{arr}", f"{op}_{arr}")
    for op in ["add", "sub"]
    for arr in ["16b", "8h", "4s", "2d", "8b", "4h", "2s"]
] + [
    (f"mul v0.{arr}, v1.{arr}, v2.{arr}", f"mul_{arr}")
    for arr in ["16b", "8h", "4s", "8b", "4h", "2s"]
]

SIMD_VEC_FP = [
    (f"{op} v0.{arr}, v1.{arr}, v2.{arr}", f"v{op}_{arr}")
    for op in ["fadd", "fsub", "fmul", "fdiv", "fmax", "fmin"]
    for arr in ["4s", "2d", "2s"]
] + [
    (f"fmla v0.{arr}, v1.{arr}, v2.{arr}", f"fmla_{arr}")
    for arr in ["4s", "2d", "2s"]
] + [
    (f"fmls v0.{arr}, v1.{arr}, v2.{arr}", f"fmls_{arr}")
    for arr in ["4s", "2d", "2s"]
]

SIMD_VEC_LOGIC = [
    (f"{op} v0.16b, v1.16b, v2.16b", f"v{op}")
    for op in ["and", "orr", "eor", "bic", "orn", "bif", "bit", "bsl"]
] + [
    ("not v0.16b, v1.16b", "vnot"),
]

SIMD_VEC_CMP = [
    (f"{op} v0.{arr}, v1.{arr}, v2.{arr}", f"v{op}_{arr}")
    for op in ["cmhi", "cmge", "cmgt", "cmeq", "cmhs"]
    for arr in ["4s", "2d", "16b"]
] + [
    (f"fcmeq v0.{arr}, v1.{arr}, v2.{arr}", f"fcmeq_{arr}")
    for arr in ["4s", "2d"]
] + [
    (f"fcmgt v0.{arr}, v1.{arr}, v2.{arr}", f"fcmgt_{arr}")
    for arr in ["4s", "2d"]
] + [
    (f"fcmge v0.{arr}, v1.{arr}, v2.{arr}", f"fcmge_{arr}")
    for arr in ["4s", "2d"]
]

SIMD_VEC_SHIFT = [
    (f"shl v0.{arr}, v1.{arr}, #{amt}", f"shl_{arr}_{amt}")
    for arr, amts in [("4s", [1, 8, 16, 31]), ("2d", [1, 32, 63]),
                      ("8h", [1, 8, 15]), ("16b", [1, 4, 7])]
    for amt in amts
] + [
    (f"ushr v0.{arr}, v1.{arr}, #{amt}", f"ushr_{arr}_{amt}")
    for arr, amts in [("4s", [1, 16, 32]), ("2d", [1, 32, 64]),
                      ("8h", [1, 8, 16])]
    for amt in amts
] + [
    (f"sshr v0.{arr}, v1.{arr}, #{amt}", f"sshr_{arr}_{amt}")
    for arr, amts in [("4s", [1, 16, 32]), ("2d", [1, 32, 64])]
    for amt in amts
]

SIMD_VEC_PERM = [
    (f"{op} v0.{arr}, v1.{arr}, v2.{arr}", f"{op}_{arr}")
    for op in ["zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2"]
    for arr in ["4s", "2d", "8h", "16b"]
] + [
    (f"ext v0.16b, v1.16b, v2.16b, #{idx}", f"ext_{idx}")
    for idx in [0, 4, 8, 15]
] + [
    (f"rev64 v0.{arr}, v1.{arr}", f"rev64_{arr}")
    for arr in ["4s", "8h", "16b"]
] + [
    (f"rev32 v0.{arr}, v1.{arr}", f"rev32_{arr}")
    for arr in ["8h", "16b"]
] + [
    (f"rev16 v0.16b, v1.16b", "rev16_16b"),
]

SIMD_VEC_MISC = [
    # DUP
    (f"dup v0.4s, w0", "dup_4s_w"),
    (f"dup v0.2d, x0", "dup_2d_x"),
    (f"dup v0.8h, w0", "dup_8h_w"),
    (f"dup v0.16b, w0", "dup_16b_w"),
    (f"dup v0.4s, v1.s[0]", "dup_4s_elem"),
    (f"dup v0.4s, v1.s[3]", "dup_4s_elem3"),
    (f"dup v0.2d, v1.d[0]", "dup_2d_elem"),
    # INS/UMOV/SMOV
    (f"ins v0.s[0], w0", "ins_s0_w0"),
    (f"ins v0.d[0], x0", "ins_d0_x0"),
    (f"ins v0.b[0], w0", "ins_b0_w0"),
    (f"ins v0.s[0], v1.s[1]", "ins_s0_v1s1"),
    (f"umov w0, v0.s[0]", "umov_w_s0"),
    (f"umov x0, v0.d[0]", "umov_x_d0"),
    (f"umov w0, v0.b[0]", "umov_w_b0"),
    (f"umov w0, v0.h[0]", "umov_w_h0"),
    (f"smov w0, v0.b[0]", "smov_w_b0"),
    (f"smov w0, v0.h[0]", "smov_w_h0"),
    (f"smov x0, v0.s[0]", "smov_x_s0"),
    # MOVI
    (f"movi v0.4s, #0", "movi_4s_0"),
    (f"movi v0.4s, #0xff", "movi_4s_ff"),
    (f"movi v0.16b, #0", "movi_16b_0"),
    (f"movi v0.16b, #0xff", "movi_16b_ff"),
    (f"movi v0.2d, #0", "movi_2d_0"),
    # FMOV vector
    (f"fmov v0.4s, #1.0", "fmov_v4s_1"),
    (f"fmov v0.2d, #1.0", "fmov_v2d_1"),
    # CNT
    (f"cnt v0.16b, v1.16b", "cnt_16b"),
    (f"cnt v0.8b, v1.8b", "cnt_8b"),
    # ABS
    (f"abs v0.4s, v1.4s", "abs_4s"),
    (f"abs v0.2d, v1.2d", "abs_2d"),
    (f"neg v0.4s, v1.4s", "vneg_4s"),
    (f"neg v0.2d, v1.2d", "vneg_2d"),
    (f"fneg v0.4s, v1.4s", "fneg_4s"),
    (f"fneg v0.2d, v1.2d", "fneg_2d"),
    (f"fabs v0.4s, v1.4s", "vfabs_4s"),
    (f"fabs v0.2d, v1.2d", "vfabs_2d"),
]

SIMD_VEC_LDST = [
    ("ldr q0, [x0]", "ldr_q"),
    ("str q0, [x0]", "str_q"),
    ("ldr q0, [x0, #16]", "ldr_q_off"),
    ("str q0, [x0, #16]", "str_q_off"),
    ("ldr q10, [sp]", "ldr_q10_sp"),
    ("ld1 {v0.4s}, [x0]", "ld1_4s"),
    ("ld1 {v0.2d}, [x0]", "ld1_2d"),
    ("ld1 {v0.16b}, [x0]", "ld1_16b"),
    ("st1 {v0.4s}, [x0]", "st1_4s"),
    ("st1 {v0.2d}, [x0]", "st1_2d"),
    ("ld1 {v0.4s, v1.4s}, [x0]", "ld1_4s_x2"),
    ("ld1 {v0.4s, v1.4s, v2.4s}, [x0]", "ld1_4s_x3"),
    ("ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]", "ld1_4s_x4"),
    ("ld2 {v0.4s, v1.4s}, [x0]", "ld2_4s"),
    ("st2 {v0.4s, v1.4s}, [x0]", "st2_4s"),
    ("ld1r {v0.4s}, [x0]", "ld1r_4s"),
    ("ld1r {v0.2d}, [x0]", "ld1r_2d"),
]

# Across lanes
SIMD_ACROSS = [
    (f"addv {sc}, v1.{arr}", f"addv_{sc}_{arr}")
    for sc, arr in [("b0", "16b"), ("h0", "8h"), ("s0", "4s")]
] + [
    (f"fmaxv s0, v1.4s", "fmaxv_4s"),
    (f"fminv s0, v1.4s", "fminv_4s"),
]

# Widen/narrow
SIMD_WIDEN = [
    ("saddl v0.4s, v1.4h, v2.4h", "saddl_4s"),
    ("uaddl v0.4s, v1.4h, v2.4h", "uaddl_4s"),
    ("saddl v0.8h, v1.8b, v2.8b", "saddl_8h"),
    ("saddl2 v0.4s, v1.8h, v2.8h", "saddl2_4s"),
    ("xtn v0.4h, v1.4s", "xtn_4h"),
    ("xtn v0.8b, v1.8h", "xtn_8b"),
    ("xtn2 v0.8h, v1.4s", "xtn2_8h"),
]

SIMD_ALL = (SIMD_VEC_ARITH + SIMD_VEC_FP + SIMD_VEC_LOGIC + SIMD_VEC_CMP +
            SIMD_VEC_SHIFT + SIMD_VEC_PERM + SIMD_VEC_MISC + SIMD_VEC_LDST +
            SIMD_ACROSS + SIMD_WIDEN)


@pytest.mark.parametrize("instr,desc", SIMD_ALL, ids=[d for _, d in SIMD_ALL])
def test_simd(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Half-precision FP (requires +fullfp16)
# ===================================================================

FP16_TESTS = [
    ("fadd h0, h1, h2", "fadd_h"),
    ("fsub h0, h1, h2", "fsub_h"),
    ("fmul h0, h1, h2", "fmul_h"),
    ("fdiv h0, h1, h2", "fdiv_h"),
    ("fabs h0, h1", "fabs_h"),
    ("fneg h0, h1", "fneg_h"),
    ("fsqrt h0, h1", "fsqrt_h"),
    ("fcmp h0, h1", "fcmp_h"),
    ("fmov h0, h1", "fmov_h"),
]


@pytest.mark.parametrize("instr,desc", FP16_TESTS, ids=[d for _, d in FP16_TESTS])
def test_fp16(asm_fullfp16, instr, desc):
    assert len(asm_fullfp16.asm(instr)) == 4


# ===================================================================
# Atomics (ARMv8.1 LSE)
# ===================================================================

LSE_TESTS = [
    (f"cas {r1}, {r2}, [{r3}]", f"cas_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("x10", "x11", "x12"),
                       ("w0", "w1", "x2"), ("w10", "w11", "x12")]
] + [
    (f"casa {r1}, {r2}, [{r3}]", f"casa_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"casal {r1}, {r2}, [{r3}]", f"casal_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"casl {r1}, {r2}, [{r3}]", f"casl_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"swp {r1}, {r2}, [{r3}]", f"swp_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"swpa {r1}, {r2}, [{r3}]", f"swpa_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"swpal {r1}, {r2}, [{r3}]", f"swpal_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldadd {r1}, {r2}, [{r3}]", f"ldadd_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"ldadda {r1}, {r2}, [{r3}]", f"ldadda_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldaddal {r1}, {r2}, [{r3}]", f"ldaddal_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldclr {r1}, {r2}, [{r3}]", f"ldclr_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"ldset {r1}, {r2}, [{r3}]", f"ldset_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2"), ("w0", "w1", "x2")]
] + [
    (f"ldsmax {r1}, {r2}, [{r3}]", f"ldsmax_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldsmin {r1}, {r2}, [{r3}]", f"ldsmin_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldumax {r1}, {r2}, [{r3}]", f"ldumax_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    (f"ldumin {r1}, {r2}, [{r3}]", f"ldumin_{r1}_{r2}_{r3}")
    for r1, r2, r3 in [("x0", "x1", "x2")]
] + [
    # Exclusive load/store (baseline)
    ("ldxr x0, [x1]", "ldxr_x"),
    ("stxr w0, x1, [x2]", "stxr_x"),
    ("ldxr w0, [x1]", "ldxr_w"),
    ("stxr w0, w1, [x2]", "stxr_w"),
    ("ldaxr x0, [x1]", "ldaxr_x"),
    ("stlxr w0, x1, [x2]", "stlxr_x"),
    ("ldaxr w0, [x1]", "ldaxr_w"),
    ("stlxr w0, w1, [x2]", "stlxr_w"),
    ("ldxrb w0, [x1]", "ldxrb"),
    ("stxrb w0, w1, [x2]", "stxrb"),
    ("ldxrh w0, [x1]", "ldxrh"),
    ("stxrh w0, w1, [x2]", "stxrh"),
    # Store-release / load-acquire
    ("ldar x0, [x1]", "ldar_x"),
    ("stlr x0, [x1]", "stlr_x"),
    ("ldar w0, [x1]", "ldar_w"),
    ("stlr w0, [x1]", "stlr_w"),
    ("ldarb w0, [x1]", "ldarb"),
    ("stlrb w0, [x1]", "stlrb"),
    ("ldarh w0, [x1]", "ldarh"),
    ("stlrh w0, [x1]", "stlrh"),
]


@pytest.mark.parametrize("instr,desc", LSE_TESTS, ids=[d for _, d in LSE_TESTS])
def test_atomics(asm_lse, instr, desc):
    assert len(asm_lse.asm(instr)) == 4


# ===================================================================
# System / barriers / hints
# ===================================================================

SYSTEM_TESTS = [
    ("svc #0", "svc_0"),
    ("svc #1", "svc_1"),
    ("svc #0xFFFF", "svc_max"),
    ("brk #0", "brk_0"),
    ("brk #1", "brk_1"),
    ("hlt #0", "hlt_0"),
    ("dmb sy", "dmb_sy"),
    ("dmb ish", "dmb_ish"),
    ("dmb ishld", "dmb_ishld"),
    ("dmb ishst", "dmb_ishst"),
    ("dmb osh", "dmb_osh"),
    ("dmb oshld", "dmb_oshld"),
    ("dmb oshst", "dmb_oshst"),
    ("dmb nsh", "dmb_nsh"),
    ("dmb nshld", "dmb_nshld"),
    ("dmb nshst", "dmb_nshst"),
    ("dsb sy", "dsb_sy"),
    ("dsb ish", "dsb_ish"),
    ("dsb ishld", "dsb_ishld"),
    ("dsb ishst", "dsb_ishst"),
    ("isb", "isb"),
    ("isb sy", "isb_sy"),
    ("nop", "nop"),
    ("wfi", "wfi"),
    ("wfe", "wfe"),
    ("sev", "sev"),
    ("sevl", "sevl"),
    ("yield", "yield"),
    ("clrex", "clrex"),
    # MRS/MSR
    ("mrs x0, NZCV", "mrs_nzcv"),
    ("msr NZCV, x0", "msr_nzcv"),
    ("mrs x0, FPCR", "mrs_fpcr"),
    ("msr FPCR, x0", "msr_fpcr"),
    ("mrs x0, FPSR", "mrs_fpsr"),
    ("msr FPSR, x0", "msr_fpsr"),
    ("mrs x0, CurrentEL", "mrs_currentel"),
    ("mrs x0, DAIF", "mrs_daif"),
    ("msr DAIF, x0", "msr_daif"),
    ("mrs x0, TPIDR_EL0", "mrs_tpidr_el0"),
    ("msr TPIDR_EL0, x0", "msr_tpidr_el0"),
    ("mrs x10, NZCV", "mrs_nzcv_x10"),
]


@pytest.mark.parametrize("instr,desc", SYSTEM_TESTS, ids=[d for _, d in SYSTEM_TESTS])
def test_system(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Address generation
# ===================================================================

ADDR_GEN_TESTS = [
    ("adr x0, target\ntarget:\nnop", "adr_fwd"),
    ("adrp x0, target\ntarget:\nnop", "adrp_fwd"),
    ("target:\nnop\nadr x0, target", "adr_back"),
    ("adr x10, target\ntarget:\nnop", "adr_x10"),
    ("adr x30, target\ntarget:\nnop", "adr_x30"),
]


@pytest.mark.parametrize("instr,desc", ADDR_GEN_TESTS, ids=[d for _, d in ADDR_GEN_TESTS])
def test_addr_gen(asm, instr, desc):
    assert len(asm.asm(instr)) > 0


# ===================================================================
# Stack patterns (multi-instruction)
# ===================================================================

STACK_TESTS = [
    ("stp x29, x30, [sp, #-16]!\nmov x29, sp", "prologue"),
    ("ldp x29, x30, [sp], #16\nret", "epilogue"),
    ("stp x19, x20, [sp, #-16]!\nstp x21, x22, [sp, #-16]!", "callee_saved"),
    ("sub sp, sp, #64\nstp x29, x30, [sp, #48]\nadd x29, sp, #48", "frame_setup"),
    ("ldp x29, x30, [sp, #48]\nadd sp, sp, #64\nret", "frame_teardown"),
    # Function with locals
    ("stp x29, x30, [sp, #-32]!\nmov x29, sp\nstr x0, [sp, #16]\nstr x1, [sp, #24]", "locals"),
]


@pytest.mark.parametrize("instr,desc", STACK_TESTS, ids=[d for _, d in STACK_TESTS])
def test_stack_patterns(asm, instr, desc):
    assert len(asm.asm(instr)) > 0


# ===================================================================
# CRC (ARMv8.1-A)
# ===================================================================

CRC_TESTS = [
    (f"{op} w0, w1, {r2}", f"{op}_{r2}")
    for op in ["crc32b", "crc32h", "crc32w"]
    for r2 in ["w2", "w10"]
] + [
    ("crc32x w0, w1, x2", "crc32x"),
    ("crc32cb w0, w1, w2", "crc32cb"),
    ("crc32ch w0, w1, w2", "crc32ch"),
    ("crc32cw w0, w1, w2", "crc32cw"),
    ("crc32cx w0, w1, x2", "crc32cx"),
]


@pytest.mark.parametrize("instr,desc", CRC_TESTS, ids=[d for _, d in CRC_TESTS])
def test_crc(asm_crc, instr, desc):
    assert len(asm_crc.asm(instr)) == 4


# ===================================================================
# MOV register (all combos)
# ===================================================================

MOV_REG_TESTS = [
    (f"mov {rd}, {rs}", f"mov_{rd}_{rs}")
    for rd, rs in X_PAIRS + W_PAIRS
] + [
    ("mov sp, x0", "mov_sp_x0"),
    ("mov x0, sp", "mov_x0_sp"),
]


@pytest.mark.parametrize("instr,desc", MOV_REG_TESTS, ids=[d for _, d in MOV_REG_TESTS])
def test_mov_reg(asm, instr, desc):
    assert len(asm.asm(instr)) == 4


# ===================================================================
# Error cases
# ===================================================================

ERROR_TESTS = [
    ("not_real x0, x1", "bad_instruction"),
    ("mov x99, x0", "bad_register"),
    ("push {r0}", "arm32_on_aarch64"),
    ("movs r0, #0", "thumb_on_aarch64"),
    ("ldr x0, [w0]", "w_reg_as_base"),
]


@pytest.mark.parametrize("instr,desc", ERROR_TESTS, ids=[d for _, d in ERROR_TESTS])
def test_errors(asm, instr, desc):
    with pytest.raises(AsmError):
        asm.asm(instr)
