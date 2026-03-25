"""Tests for x86 and x86_64 assembly.

Comprehensive corpus covering GPR ops, SIMD, control flow, string ops,
bit manipulation, system instructions, and addressing modes.
"""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def x64():
    return Assembler.x86_64()


@pytest.fixture(scope="module")
def x86():
    return Assembler.i686()


# ===================================================================
# Helpers
# ===================================================================

# All 16 GPRs
REGS64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
REGS32 = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
          "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"]
REGS16 = ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
          "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"]
REGS8 = ["al", "bl", "cl", "dl", "sil", "dil", "bpl", "spl",
         "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"]

XMM_REGS = [f"xmm{i}" for i in range(16)]
YMM_REGS = [f"ymm{i}" for i in range(16)]


# ===================================================================
# x86_64 — MOV variants (all register widths)
# ===================================================================

MOV64 = [(f"mov {r}, rax", f"mov_{r}_rax") for r in REGS64 if r != "rax"]
MOV32 = [(f"mov {r}, eax", f"mov_{r}_eax") for r in REGS32 if r != "eax"]
MOV16 = [(f"mov {r}, ax", f"mov_{r}_ax") for r in REGS16 if r != "ax"]
MOV8 = [(f"mov {r}, al", f"mov_{r}_al") for r in REGS8 if r != "al"]


@pytest.mark.parametrize("instr,desc", MOV64, ids=[d for _, d in MOV64])
def test_mov64(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", MOV32, ids=[d for _, d in MOV32])
def test_mov32(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", MOV16, ids=[d for _, d in MOV16])
def test_mov16(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", MOV8, ids=[d for _, d in MOV8])
def test_mov8(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — ALU: every op x every 64-bit register pair
# ===================================================================

ALU_OPS = ["add", "sub", "and", "or", "xor", "cmp", "test", "adc", "sbb"]
ALU_REG_PAIRS = [("rax", "rbx"), ("rcx", "rdx"), ("rsi", "rdi"),
                 ("r8", "r9"), ("r10", "r11"), ("r12", "r13"), ("r14", "r15"),
                 ("rax", "r15"), ("r8", "rax")]

ALU_TESTS = [(f"{op} {a}, {b}", f"{op}_{a}_{b}")
             for op in ALU_OPS for a, b in ALU_REG_PAIRS]


@pytest.mark.parametrize("instr,desc", ALU_TESTS, ids=[d for _, d in ALU_TESTS])
def test_alu_reg_reg(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ALU with immediates
ALU_IMM_TESTS = [
    (f"{op} {reg}, {imm}", f"{op}_{reg}_{imm:#x}")
    for op in ["add", "sub", "and", "or", "xor", "cmp"]
    for reg in ["rax", "rbx", "r8", "r15"]
    for imm in [0, 1, 0x7F, 0x80, 0xFF, 0x1000, 0x7FFFFFFF]
]


@pytest.mark.parametrize("instr,desc", ALU_IMM_TESTS, ids=[d for _, d in ALU_IMM_TESTS])
def test_alu_imm(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Shifts and rotates
# ===================================================================

SHIFT_OPS = ["shl", "shr", "sar", "rol", "ror", "rcl", "rcr"]
SHIFT_TESTS = [
    (f"{op} {reg}, {count}", f"{op}_{reg}_{count}")
    for op in SHIFT_OPS
    for reg in ["rax", "rbx", "r8", "r15"]
    for count in [1, "cl", 4, 7, 16, 31, 63]
]


@pytest.mark.parametrize("instr,desc", SHIFT_TESTS, ids=[d for _, d in SHIFT_TESTS])
def test_shifts(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Multiply and divide
# ===================================================================

MULDIV_TESTS = [
    # Unsigned multiply
    ("mul rbx", "mul_rbx"),
    ("mul r8", "mul_r8"),
    ("mul ecx", "mul_ecx"),
    # Signed multiply (1 operand)
    ("imul rbx", "imul_rbx"),
    ("imul r15", "imul_r15"),
    # Signed multiply (2 operand)
    ("imul rax, rbx", "imul_rax_rbx"),
    ("imul r8, r9", "imul_r8_r9"),
    # Signed multiply (3 operand)
    ("imul rax, rbx, 42", "imul_rax_rbx_42"),
    ("imul r8, r9, 0x100", "imul_r8_r9_0x100"),
    # Divide
    ("div rbx", "div_rbx"),
    ("div r8", "div_r8"),
    ("idiv rcx", "idiv_rcx"),
    ("idiv r15", "idiv_r15"),
    # 32-bit variants
    ("mul ebx", "mul_ebx_32"),
    ("imul eax, ebx", "imul_eax_ebx_32"),
    ("div ecx", "div_ecx_32"),
]


@pytest.mark.parametrize("instr,desc", MULDIV_TESTS, ids=[d for _, d in MULDIV_TESTS])
def test_muldiv(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Unary ALU ops
# ===================================================================

UNARY_OPS = ["inc", "dec", "neg", "not"]
UNARY_TESTS = [(f"{op} {r}", f"{op}_{r}")
               for op in UNARY_OPS
               for r in ["rax", "rbx", "r8", "r15", "eax", "r8d", "ax", "al"]]


@pytest.mark.parametrize("instr,desc", UNARY_TESTS, ids=[d for _, d in UNARY_TESTS])
def test_unary(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — MOV immediate (various sizes)
# ===================================================================

MOV_IMM_TESTS = [
    ("mov rax, 0", "mov_rax_0"),
    ("mov rax, 1", "mov_rax_1"),
    ("mov rax, 0x7FFFFFFF", "mov_rax_s32max"),
    ("mov rax, 0xFFFFFFFF", "mov_rax_u32max"),
    ("mov rax, 0x100000000", "mov_rax_above32"),
    ("mov rax, 0xDEADBEEFCAFEBABE", "mov_rax_64bit"),
    ("mov eax, 0", "mov_eax_0"),
    ("mov eax, 0xFFFFFFFF", "mov_eax_u32max"),
    ("mov ax, 0x1234", "mov_ax_16"),
    ("mov al, 0xFF", "mov_al_8"),
    ("mov r8, 0xDEADBEEF", "mov_r8_imm"),
    ("mov r15d, 42", "mov_r15d_imm"),
]


@pytest.mark.parametrize("instr,desc", MOV_IMM_TESTS, ids=[d for _, d in MOV_IMM_TESTS])
def test_mov_imm(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Push / Pop (all registers)
# ===================================================================

PUSHPOP_REGS = [r for r in REGS64 if r != "rsp"]  # push rsp is valid but weird
PUSH_TESTS = [(f"push {r}", f"push_{r}") for r in PUSHPOP_REGS]
POP_TESTS = [(f"pop {r}", f"pop_{r}") for r in PUSHPOP_REGS]


@pytest.mark.parametrize("instr,desc", PUSH_TESTS, ids=[d for _, d in PUSH_TESTS])
def test_push(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", POP_TESTS, ids=[d for _, d in POP_TESTS])
def test_pop(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — LEA addressing modes
# ===================================================================

LEA_TESTS = [
    ("lea rax, [rbx]", "base_only"),
    ("lea rax, [rbx + 8]", "base_disp8"),
    ("lea rax, [rbx + 0x1000]", "base_disp32"),
    ("lea rax, [rbx + rcx]", "base_index"),
    ("lea rax, [rbx + rcx*2]", "base_index_scale2"),
    ("lea rax, [rbx + rcx*4]", "base_index_scale4"),
    ("lea rax, [rbx + rcx*8]", "base_index_scale8"),
    ("lea rax, [rbx + rcx*4 + 8]", "base_index_scale_disp"),
    ("lea rax, [rcx*4 + 0x100]", "index_scale_disp"),
    ("lea rax, [rip + 0x10]", "rip_relative"),
    ("lea r8, [r9 + r10*8 + 0x20]", "rex_full"),
    ("lea eax, [ebx + ecx*4]", "addr32"),
    # All base registers
    *[(f"lea rax, [{r}]", f"base_{r}") for r in REGS64 if r != "rsp"],
]


@pytest.mark.parametrize("instr,desc", LEA_TESTS, ids=[d for _, d in LEA_TESTS])
def test_lea(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Load / Store (memory operands)
# ===================================================================

MEMOP_TESTS = [
    # MOV load
    ("mov rax, [rbx]", "load64"),
    ("mov eax, [rbx]", "load32"),
    ("mov ax, [rbx]", "load16"),
    ("mov al, [rbx]", "load8"),
    # MOV store
    ("mov [rbx], rax", "store64"),
    ("mov [rbx], eax", "store32"),
    ("mov [rbx], ax", "store16"),
    ("mov [rbx], al", "store8"),
    # With displacements
    ("mov rax, [rbx + 4]", "load_disp8"),
    ("mov rax, [rbx + 0x1000]", "load_disp32"),
    ("mov rax, [rbx + rcx*8]", "load_sib"),
    ("mov rax, [rbx + rcx*8 + 0x10]", "load_sib_disp"),
    # RIP-relative
    ("mov rax, [rip + 0x100]", "load_rip"),
    # Zero/sign extend
    ("movzx rax, byte ptr [rbx]", "movzx_byte"),
    ("movzx rax, word ptr [rbx]", "movzx_word"),
    ("movsx rax, byte ptr [rbx]", "movsx_byte"),
    ("movsx rax, word ptr [rbx]", "movsx_word"),
    ("movsxd rax, dword ptr [rbx]", "movsxd"),
    # XCHG
    ("xchg rax, rbx", "xchg_rax_rbx"),
    ("xchg [rbx], rax", "xchg_mem"),
    # CMPXCHG
    ("cmpxchg [rbx], rcx", "cmpxchg"),
    ("lock cmpxchg [rbx], rcx", "lock_cmpxchg"),
    # XADD
    ("lock xadd [rbx], rax", "lock_xadd"),
    # Byte swap
    ("bswap rax", "bswap64"),
    ("bswap eax", "bswap32"),
]


@pytest.mark.parametrize("instr,desc", MEMOP_TESTS, ids=[d for _, d in MEMOP_TESTS])
def test_memops(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Conditional moves
# ===================================================================

CMOV_CONDITIONS = [
    "cmova", "cmovae", "cmovb", "cmovbe", "cmovc",
    "cmove", "cmovg", "cmovge", "cmovl", "cmovle",
    "cmovna", "cmovnae", "cmovnb", "cmovnbe", "cmovnc",
    "cmovne", "cmovng", "cmovnge", "cmovnl", "cmovnle",
    "cmovno", "cmovnp", "cmovns", "cmovnz",
    "cmovo", "cmovp", "cmovs", "cmovz",
]

CMOV_TESTS = [(f"{cc} rax, rbx", f"{cc}_rax_rbx") for cc in CMOV_CONDITIONS]


@pytest.mark.parametrize("instr,desc", CMOV_TESTS, ids=[d for _, d in CMOV_TESTS])
def test_cmov(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — SETcc
# ===================================================================

SETCC_CONDITIONS = [
    "seta", "setae", "setb", "setbe", "setc",
    "sete", "setg", "setge", "setl", "setle",
    "setna", "setnae", "setnb", "setnbe", "setnc",
    "setne", "setng", "setnge", "setnl", "setnle",
    "setno", "setnp", "setns", "setnz",
    "seto", "setp", "sets", "setz",
]

SETCC_TESTS = [(f"{cc} al", f"{cc}_al") for cc in SETCC_CONDITIONS]


@pytest.mark.parametrize("instr,desc", SETCC_TESTS, ids=[d for _, d in SETCC_TESTS])
def test_setcc(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Jcc (conditional jumps)
# ===================================================================

JCC_CONDITIONS = [
    "ja", "jae", "jb", "jbe", "jc",
    "je", "jg", "jge", "jl", "jle",
    "jna", "jnae", "jnb", "jnbe", "jnc",
    "jne", "jng", "jnge", "jnl", "jnle",
    "jno", "jnp", "jns", "jnz",
    "jo", "jp", "js", "jz",
    "jecxz", "jrcxz",
]

JCC_TESTS = [(f"here_{cc}:\n{cc} here_{cc}", f"{cc}") for cc in JCC_CONDITIONS]


@pytest.mark.parametrize("instr,desc", JCC_TESTS, ids=[d for _, d in JCC_TESTS])
def test_jcc(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Control flow
# ===================================================================

CONTROL_FLOW_TESTS = [
    ("ret", "ret"),
    ("nop", "nop"),
    ("here_jmp:\njmp here_jmp", "jmp_self"),
    ("here_call:\ncall here_call", "call_self"),
    ("jmp rax", "jmp_reg"),
    ("call rax", "call_reg"),
    ("jmp [rax]", "jmp_mem"),
    ("call [rax]", "call_mem"),
    ("int 3", "int3"),
    ("int 0x80", "int_0x80"),
    ("syscall", "syscall"),
    ("sysenter", "sysenter"),
    ("ud2", "ud2"),
    ("hlt", "hlt"),
    ("leave", "leave"),
    ("enter 0, 0", "enter"),
    # Loop variants
    ("here_loop:\nloop here_loop", "loop"),
    ("here_loope:\nloope here_loope", "loope"),
    ("here_loopne:\nloopne here_loopne", "loopne"),
]


@pytest.mark.parametrize("instr,desc", CONTROL_FLOW_TESTS, ids=[d for _, d in CONTROL_FLOW_TESTS])
def test_control_flow(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — String operations
# ===================================================================

STRING_OPS = [
    ("movsb", "movsb"),
    ("movsw", "movsw"),
    ("movsd", "movsd_str"),
    ("movsq", "movsq"),
    ("stosb", "stosb"),
    ("stosw", "stosw"),
    ("stosd", "stosd"),
    ("stosq", "stosq"),
    ("lodsb", "lodsb"),
    ("lodsw", "lodsw"),
    ("lodsd", "lodsd"),
    ("lodsq", "lodsq"),
    ("scasb", "scasb"),
    ("scasw", "scasw"),
    ("scasd", "scasd"),
    ("scasq", "scasq"),
    ("cmpsb", "cmpsb"),
    ("cmpsw", "cmpsw"),
    ("cmpsd", "cmpsd_str"),
    ("cmpsq", "cmpsq"),
    # REP prefixed
    ("rep movsb", "rep_movsb"),
    ("rep movsq", "rep_movsq"),
    ("rep stosb", "rep_stosb"),
    ("rep stosq", "rep_stosq"),
    ("repe cmpsb", "repe_cmpsb"),
    ("repne scasb", "repne_scasb"),
]


@pytest.mark.parametrize("instr,desc", STRING_OPS, ids=[d for _, d in STRING_OPS])
def test_string_ops(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Flag manipulation
# ===================================================================

FLAG_OPS = [
    ("clc", "clc"), ("stc", "stc"), ("cmc", "cmc"),
    ("cld", "cld"), ("std", "std"),
    ("lahf", "lahf"), ("sahf", "sahf"),
    ("pushfq", "pushfq"), ("popfq", "popfq"),
]


@pytest.mark.parametrize("instr,desc", FLAG_OPS, ids=[d for _, d in FLAG_OPS])
def test_flag_ops(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Bit manipulation (BMI1/BMI2, POPCNT, LZCNT, TZCNT)
# ===================================================================

BIT_MANIP_TESTS = [
    # BSF/BSR
    ("bsf rax, rbx", "bsf"),
    ("bsr rax, rbx", "bsr"),
    # BT/BTS/BTR/BTC
    ("bt rax, rbx", "bt_reg"),
    ("bt rax, 7", "bt_imm"),
    ("bts rax, rbx", "bts"),
    ("btr rax, rbx", "btr"),
    ("btc rax, rbx", "btc"),
    ("bts [rbx], rax", "bts_mem"),
    # POPCNT
    ("popcnt rax, rbx", "popcnt64"),
    ("popcnt eax, ebx", "popcnt32"),
    # LZCNT
    ("lzcnt rax, rbx", "lzcnt64"),
    ("lzcnt eax, ebx", "lzcnt32"),
    # TZCNT
    ("tzcnt rax, rbx", "tzcnt64"),
    ("tzcnt eax, ebx", "tzcnt32"),
    # BMI1
    ("andn rax, rbx, rcx", "andn"),
    ("bextr rax, rbx, rcx", "bextr"),
    ("blsi rax, rbx", "blsi"),
    ("blsmsk rax, rbx", "blsmsk"),
    ("blsr rax, rbx", "blsr"),
    # BMI2
    ("bzhi rax, rbx, rcx", "bzhi"),
    ("pdep rax, rbx, rcx", "pdep"),
    ("pext rax, rbx, rcx", "pext"),
    ("mulx rdx, rax, rbx", "mulx"),
    ("rorx rax, rbx, 3", "rorx"),
    ("sarx rax, rbx, rcx", "sarx"),
    ("shlx rax, rbx, rcx", "shlx"),
    ("shrx rax, rbx, rcx", "shrx"),
]


@pytest.mark.parametrize("instr,desc", BIT_MANIP_TESTS, ids=[d for _, d in BIT_MANIP_TESTS])
def test_bit_manip(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Conversion / sign-extend
# ===================================================================

CONVERT_TESTS = [
    ("cbw", "cbw"),
    ("cwde", "cwde"),
    ("cdqe", "cdqe"),
    ("cwd", "cwd"),
    ("cdq", "cdq"),
    ("cqo", "cqo"),
    ("movzx eax, al", "movzx_reg_8"),
    ("movzx eax, ax", "movzx_reg_16"),
    ("movsx rax, al", "movsx_reg_8"),
    ("movsx rax, ax", "movsx_reg_16"),
    ("movsxd rax, eax", "movsxd_reg"),
]


@pytest.mark.parametrize("instr,desc", CONVERT_TESTS, ids=[d for _, d in CONVERT_TESTS])
def test_conversions(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — SSE (all 16 XMM registers)
# ===================================================================

SSE_ARITH = ["addss", "subss", "mulss", "divss", "sqrtss", "maxss", "minss",
             "addsd", "subsd", "mulsd", "divsd", "sqrtsd", "maxsd", "minsd",
             "addps", "subps", "mulps", "divps", "sqrtps", "maxps", "minps",
             "addpd", "subpd", "mulpd", "divpd", "sqrtpd", "maxpd", "minpd"]

SSE_ARITH_TESTS = [(f"{op} xmm0, xmm1", f"{op}_xmm0_xmm1") for op in SSE_ARITH]

# Cross-register SSE: every xmm pair (0,N) for packed add
SSE_XMM_TESTS = [(f"addps xmm0, {r}", f"addps_xmm0_{r}") for r in XMM_REGS[1:]]


@pytest.mark.parametrize("instr,desc", SSE_ARITH_TESTS, ids=[d for _, d in SSE_ARITH_TESTS])
def test_sse_arith(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", SSE_XMM_TESTS, ids=[d for _, d in SSE_XMM_TESTS])
def test_sse_xmm_regs(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


SSE_MOV_TESTS = [
    ("movaps xmm0, xmm1", "movaps"),
    ("movups xmm0, xmm1", "movups"),
    ("movapd xmm0, xmm1", "movapd"),
    ("movupd xmm0, xmm1", "movupd"),
    ("movss xmm0, xmm1", "movss"),
    ("movsd xmm0, xmm1", "movsd"),
    ("movhlps xmm0, xmm1", "movhlps"),
    ("movlhps xmm0, xmm1", "movlhps"),
    ("movdqa xmm0, xmm1", "movdqa"),
    ("movdqu xmm0, xmm1", "movdqu"),
    ("movd eax, xmm0", "movd_gpr_xmm"),
    ("movd xmm0, eax", "movd_xmm_gpr"),
    ("movq rax, xmm0", "movq_gpr_xmm"),
    ("movq xmm0, rax", "movq_xmm_gpr"),
    # Shuffles
    ("shufps xmm0, xmm1, 0x1B", "shufps"),
    ("shufpd xmm0, xmm1, 1", "shufpd"),
    ("pshufd xmm0, xmm1, 0xE4", "pshufd"),
    ("pshufhw xmm0, xmm1, 0xE4", "pshufhw"),
    ("pshuflw xmm0, xmm1, 0xE4", "pshuflw"),
    ("unpcklps xmm0, xmm1", "unpcklps"),
    ("unpckhps xmm0, xmm1", "unpckhps"),
    # Conversions
    ("cvtsi2ss xmm0, eax", "cvtsi2ss_32"),
    ("cvtsi2ss xmm0, rax", "cvtsi2ss_64"),
    ("cvtsi2sd xmm0, eax", "cvtsi2sd_32"),
    ("cvtsi2sd xmm0, rax", "cvtsi2sd_64"),
    ("cvtss2si eax, xmm0", "cvtss2si_32"),
    ("cvtss2si rax, xmm0", "cvtss2si_64"),
    ("cvtsd2si eax, xmm0", "cvtsd2si_32"),
    ("cvtsd2si rax, xmm0", "cvtsd2si_64"),
    ("cvtss2sd xmm0, xmm1", "cvtss2sd"),
    ("cvtsd2ss xmm0, xmm1", "cvtsd2ss"),
    ("cvtps2pd xmm0, xmm1", "cvtps2pd"),
    ("cvtpd2ps xmm0, xmm1", "cvtpd2ps"),
    ("cvttss2si eax, xmm0", "cvttss2si"),
    ("cvttsd2si rax, xmm0", "cvttsd2si"),
]


@pytest.mark.parametrize("instr,desc", SSE_MOV_TESTS, ids=[d for _, d in SSE_MOV_TESTS])
def test_sse_mov(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


SSE_CMP_TESTS = [
    ("comiss xmm0, xmm1", "comiss"),
    ("comisd xmm0, xmm1", "comisd"),
    ("ucomiss xmm0, xmm1", "ucomiss"),
    ("ucomisd xmm0, xmm1", "ucomisd"),
    ("cmpps xmm0, xmm1, 0", "cmpps_eq"),
    ("cmpps xmm0, xmm1, 1", "cmpps_lt"),
    ("cmpps xmm0, xmm1, 2", "cmpps_le"),
    ("cmppd xmm0, xmm1, 0", "cmppd_eq"),
    ("cmpss xmm0, xmm1, 0", "cmpss_eq"),
    ("cmpsd xmm0, xmm1, 0", "cmpsd_eq"),
]


@pytest.mark.parametrize("instr,desc", SSE_CMP_TESTS, ids=[d for _, d in SSE_CMP_TESTS])
def test_sse_cmp(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — SSE2 integer SIMD
# ===================================================================

SSE2_INT_TESTS = [
    ("paddb xmm0, xmm1", "paddb"),
    ("paddw xmm0, xmm1", "paddw"),
    ("paddd xmm0, xmm1", "paddd"),
    ("paddq xmm0, xmm1", "paddq"),
    ("psubb xmm0, xmm1", "psubb"),
    ("psubw xmm0, xmm1", "psubw"),
    ("psubd xmm0, xmm1", "psubd"),
    ("psubq xmm0, xmm1", "psubq"),
    ("pmullw xmm0, xmm1", "pmullw"),
    ("pmulhw xmm0, xmm1", "pmulhw"),
    ("pmuludq xmm0, xmm1", "pmuludq"),
    ("pand xmm0, xmm1", "pand"),
    ("pandn xmm0, xmm1", "pandn"),
    ("por xmm0, xmm1", "por"),
    ("pxor xmm0, xmm1", "pxor"),
    ("pcmpeqb xmm0, xmm1", "pcmpeqb"),
    ("pcmpeqw xmm0, xmm1", "pcmpeqw"),
    ("pcmpeqd xmm0, xmm1", "pcmpeqd"),
    ("pcmpgtb xmm0, xmm1", "pcmpgtb"),
    ("pcmpgtw xmm0, xmm1", "pcmpgtw"),
    ("pcmpgtd xmm0, xmm1", "pcmpgtd"),
    ("packuswb xmm0, xmm1", "packuswb"),
    ("packsswb xmm0, xmm1", "packsswb"),
    ("packssdw xmm0, xmm1", "packssdw"),
    ("punpcklbw xmm0, xmm1", "punpcklbw"),
    ("punpckhbw xmm0, xmm1", "punpckhbw"),
    ("punpcklwd xmm0, xmm1", "punpcklwd"),
    ("punpckhwd xmm0, xmm1", "punpckhwd"),
    ("punpckldq xmm0, xmm1", "punpckldq"),
    ("punpckhdq xmm0, xmm1", "punpckhdq"),
    ("punpcklqdq xmm0, xmm1", "punpcklqdq"),
    ("punpckhqdq xmm0, xmm1", "punpckhqdq"),
    ("psllw xmm0, 4", "psllw_imm"),
    ("pslld xmm0, 4", "pslld_imm"),
    ("psllq xmm0, 4", "psllq_imm"),
    ("psrlw xmm0, 4", "psrlw_imm"),
    ("psrld xmm0, 4", "psrld_imm"),
    ("psrlq xmm0, 4", "psrlq_imm"),
    ("psraw xmm0, 4", "psraw_imm"),
    ("psrad xmm0, 4", "psrad_imm"),
    # Saturating
    ("paddsb xmm0, xmm1", "paddsb"),
    ("paddsw xmm0, xmm1", "paddsw"),
    ("paddusb xmm0, xmm1", "paddusb"),
    ("paddusw xmm0, xmm1", "paddusw"),
    ("psubsb xmm0, xmm1", "psubsb"),
    ("psubsw xmm0, xmm1", "psubsw"),
    ("psubusb xmm0, xmm1", "psubusb"),
    ("psubusw xmm0, xmm1", "psubusw"),
    # Min/Max
    ("pminsw xmm0, xmm1", "pminsw"),
    ("pminub xmm0, xmm1", "pminub"),
    ("pmaxsw xmm0, xmm1", "pmaxsw"),
    ("pmaxub xmm0, xmm1", "pmaxub"),
    # Average
    ("pavgb xmm0, xmm1", "pavgb"),
    ("pavgw xmm0, xmm1", "pavgw"),
    # SAD
    ("psadbw xmm0, xmm1", "psadbw"),
    # Extract/Insert
    ("pextrw eax, xmm0, 3", "pextrw"),
    ("pinsrw xmm0, eax, 3", "pinsrw"),
    # Mask
    ("pmovmskb eax, xmm0", "pmovmskb"),
    ("movmskps eax, xmm0", "movmskps"),
    ("movmskpd eax, xmm0", "movmskpd"),
]


@pytest.mark.parametrize("instr,desc", SSE2_INT_TESTS, ids=[d for _, d in SSE2_INT_TESTS])
def test_sse2_int(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — SSE3/SSSE3/SSE4.1/SSE4.2
# ===================================================================

SSE3PLUS_TESTS = [
    # SSE3
    ("haddps xmm0, xmm1", "haddps"),
    ("hsubps xmm0, xmm1", "hsubps"),
    ("haddpd xmm0, xmm1", "haddpd"),
    ("hsubpd xmm0, xmm1", "hsubpd"),
    ("addsubps xmm0, xmm1", "addsubps"),
    ("addsubpd xmm0, xmm1", "addsubpd"),
    ("movshdup xmm0, xmm1", "movshdup"),
    ("movsldup xmm0, xmm1", "movsldup"),
    ("movddup xmm0, xmm1", "movddup"),
    # SSSE3
    ("pshufb xmm0, xmm1", "pshufb"),
    ("phaddw xmm0, xmm1", "phaddw"),
    ("phaddd xmm0, xmm1", "phaddd"),
    ("phsubw xmm0, xmm1", "phsubw"),
    ("phsubd xmm0, xmm1", "phsubd"),
    ("pmaddubsw xmm0, xmm1", "pmaddubsw"),
    ("pabsb xmm0, xmm1", "pabsb"),
    ("pabsw xmm0, xmm1", "pabsw"),
    ("pabsd xmm0, xmm1", "pabsd"),
    ("palignr xmm0, xmm1, 4", "palignr"),
    ("psignb xmm0, xmm1", "psignb"),
    ("psignw xmm0, xmm1", "psignw"),
    ("psignd xmm0, xmm1", "psignd"),
    ("pmulhrsw xmm0, xmm1", "pmulhrsw"),
    # SSE4.1
    ("pmulld xmm0, xmm1", "pmulld"),
    ("pmuldq xmm0, xmm1", "pmuldq"),
    ("pminsd xmm0, xmm1", "pminsd"),
    ("pminud xmm0, xmm1", "pminud"),
    ("pminsb xmm0, xmm1", "pminsb"),
    ("pminuw xmm0, xmm1", "pminuw"),
    ("pmaxsd xmm0, xmm1", "pmaxsd"),
    ("pmaxud xmm0, xmm1", "pmaxud"),
    ("pmaxsb xmm0, xmm1", "pmaxsb"),
    ("pmaxuw xmm0, xmm1", "pmaxuw"),
    ("ptest xmm0, xmm1", "ptest"),
    ("roundps xmm0, xmm1, 0", "roundps"),
    ("roundpd xmm0, xmm1, 0", "roundpd"),
    ("roundss xmm0, xmm1, 0", "roundss"),
    ("roundsd xmm0, xmm1, 0", "roundsd"),
    ("blendps xmm0, xmm1, 0x5", "blendps"),
    ("blendpd xmm0, xmm1, 1", "blendpd"),
    ("pblendw xmm0, xmm1, 0xAA", "pblendw"),
    ("blendvps xmm0, xmm1, xmm0", "blendvps"),
    ("blendvpd xmm0, xmm1, xmm0", "blendvpd"),
    ("pblendvb xmm0, xmm1, xmm0", "pblendvb"),
    ("dpps xmm0, xmm1, 0xFF", "dpps"),
    ("dppd xmm0, xmm1, 0x31", "dppd"),
    ("insertps xmm0, xmm1, 0x10", "insertps"),
    ("extractps eax, xmm0, 1", "extractps"),
    ("pinsrb xmm0, eax, 3", "pinsrb"),
    ("pinsrd xmm0, eax, 1", "pinsrd"),
    ("pinsrq xmm0, rax, 0", "pinsrq"),
    ("pextrb eax, xmm0, 3", "pextrb"),
    ("pextrd eax, xmm0, 1", "pextrd"),
    ("pextrq rax, xmm0, 0", "pextrq"),
    ("pcmpeqq xmm0, xmm1", "pcmpeqq"),
    ("packusdw xmm0, xmm1", "packusdw"),
    ("pmovzxbw xmm0, xmm1", "pmovzxbw"),
    ("pmovzxbd xmm0, xmm1", "pmovzxbd"),
    ("pmovzxbq xmm0, xmm1", "pmovzxbq"),
    ("pmovzxwd xmm0, xmm1", "pmovzxwd"),
    ("pmovzxwq xmm0, xmm1", "pmovzxwq"),
    ("pmovzxdq xmm0, xmm1", "pmovzxdq"),
    ("pmovsxbw xmm0, xmm1", "pmovsxbw"),
    ("pmovsxbd xmm0, xmm1", "pmovsxbd"),
    ("pmovsxbq xmm0, xmm1", "pmovsxbq"),
    ("pmovsxwd xmm0, xmm1", "pmovsxwd"),
    ("pmovsxwq xmm0, xmm1", "pmovsxwq"),
    ("pmovsxdq xmm0, xmm1", "pmovsxdq"),
    ("mpsadbw xmm0, xmm1, 0", "mpsadbw"),
    ("phminposuw xmm0, xmm1", "phminposuw"),
    # SSE4.2
    ("pcmpgtq xmm0, xmm1", "pcmpgtq"),
    ("pcmpestri xmm0, xmm1, 0", "pcmpestri"),
    ("pcmpestrm xmm0, xmm1, 0", "pcmpestrm"),
    ("pcmpistri xmm0, xmm1, 0", "pcmpistri"),
    ("pcmpistrm xmm0, xmm1, 0", "pcmpistrm"),
    ("crc32 eax, bl", "crc32_8"),
    ("crc32 eax, bx", "crc32_16"),
    ("crc32 eax, ebx", "crc32_32"),
    ("crc32 rax, bl", "crc32_64_8"),
    ("crc32 rax, rbx", "crc32_64_64"),
]


@pytest.mark.parametrize("instr,desc", SSE3PLUS_TESTS, ids=[d for _, d in SSE3PLUS_TESTS])
def test_sse3plus(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — AVX (3-operand VEX encoding)
# ===================================================================

AVX_ARITH = [
    "vaddps", "vsubps", "vmulps", "vdivps", "vmaxps", "vminps",
    "vaddpd", "vsubpd", "vmulpd", "vdivpd", "vmaxpd", "vminpd",
    "vaddss", "vsubss", "vmulss", "vdivss",
    "vaddsd", "vsubsd", "vmulsd", "vdivsd",
    "vandps", "vandnps", "vorps", "vxorps",
    "vandpd", "vandnpd", "vorpd", "vxorpd",
]

AVX_XMM_TESTS = [(f"{op} xmm0, xmm1, xmm2", f"{op}_xmm") for op in AVX_ARITH]
AVX_YMM_TESTS = [(f"{op} ymm0, ymm1, ymm2", f"{op}_ymm")
                  for op in AVX_ARITH if "ss" not in op and "sd" not in op]


@pytest.mark.parametrize("instr,desc", AVX_XMM_TESTS, ids=[d for _, d in AVX_XMM_TESTS])
def test_avx_xmm(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


@pytest.mark.parametrize("instr,desc", AVX_YMM_TESTS, ids=[d for _, d in AVX_YMM_TESTS])
def test_avx_ymm(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# Cross-register AVX: every ymm pair (0,N)
AVX_YMM_CROSS = [(f"vaddps ymm0, ymm1, {r}", f"vaddps_ymm0_ymm1_{r}")
                  for r in YMM_REGS[2:]]


@pytest.mark.parametrize("instr,desc", AVX_YMM_CROSS, ids=[d for _, d in AVX_YMM_CROSS])
def test_avx_ymm_regs(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


AVX_MISC_TESTS = [
    ("vsqrtps xmm0, xmm1", "vsqrtps_xmm"),
    ("vsqrtps ymm0, ymm1", "vsqrtps_ymm"),
    ("vsqrtpd xmm0, xmm1", "vsqrtpd_xmm"),
    ("vsqrtpd ymm0, ymm1", "vsqrtpd_ymm"),
    ("vsqrtss xmm0, xmm1, xmm2", "vsqrtss"),
    ("vsqrtsd xmm0, xmm1, xmm2", "vsqrtsd"),
    ("vrcpps xmm0, xmm1", "vrcpps_xmm"),
    ("vrcpps ymm0, ymm1", "vrcpps_ymm"),
    ("vrsqrtps xmm0, xmm1", "vrsqrtps_xmm"),
    ("vrsqrtps ymm0, ymm1", "vrsqrtps_ymm"),
    ("vmovaps xmm0, xmm1", "vmovaps_xmm"),
    ("vmovaps ymm0, ymm1", "vmovaps_ymm"),
    ("vmovups xmm0, xmm1", "vmovups_xmm"),
    ("vmovups ymm0, ymm1", "vmovups_ymm"),
    ("vmovdqa xmm0, xmm1", "vmovdqa_xmm"),
    ("vmovdqa ymm0, ymm1", "vmovdqa_ymm"),
    ("vmovdqu xmm0, xmm1", "vmovdqu_xmm"),
    ("vmovdqu ymm0, ymm1", "vmovdqu_ymm"),
    ("vbroadcastss xmm0, xmm1", "vbroadcastss_xmm"),
    ("vbroadcastss ymm0, xmm1", "vbroadcastss_ymm"),
    ("vbroadcastsd ymm0, xmm1", "vbroadcastsd_ymm"),
    ("vperm2f128 ymm0, ymm1, ymm2, 0x31", "vperm2f128"),
    ("vpermilps xmm0, xmm1, 0xE4", "vpermilps_imm"),
    ("vpermilpd xmm0, xmm1, 1", "vpermilpd_imm"),
    ("vshufps xmm0, xmm1, xmm2, 0x1B", "vshufps"),
    ("vshufpd xmm0, xmm1, xmm2, 1", "vshufpd"),
    ("vunpcklps xmm0, xmm1, xmm2", "vunpcklps"),
    ("vunpckhps xmm0, xmm1, xmm2", "vunpckhps"),
    ("vblendps xmm0, xmm1, xmm2, 5", "vblendps"),
    ("vblendpd ymm0, ymm1, ymm2, 5", "vblendpd_ymm"),
    ("vroundps xmm0, xmm1, 0", "vroundps"),
    ("vroundpd ymm0, ymm1, 0", "vroundpd_ymm"),
    ("vdpps xmm0, xmm1, xmm2, 0xFF", "vdpps"),
    ("vdpps ymm0, ymm1, ymm2, 0xFF", "vdpps_ymm"),
    ("vinsertf128 ymm0, ymm1, xmm2, 1", "vinsertf128"),
    ("vextractf128 xmm0, ymm1, 1", "vextractf128"),
    ("vzeroall", "vzeroall"),
    ("vzeroupper", "vzeroupper"),
    # FMA3
    ("vfmadd132ps xmm0, xmm1, xmm2", "vfmadd132ps_xmm"),
    ("vfmadd213ps xmm0, xmm1, xmm2", "vfmadd213ps_xmm"),
    ("vfmadd231ps xmm0, xmm1, xmm2", "vfmadd231ps_xmm"),
    ("vfmadd132ps ymm0, ymm1, ymm2", "vfmadd132ps_ymm"),
    ("vfmadd213ps ymm0, ymm1, ymm2", "vfmadd213ps_ymm"),
    ("vfmadd231ps ymm0, ymm1, ymm2", "vfmadd231ps_ymm"),
    ("vfmadd132pd xmm0, xmm1, xmm2", "vfmadd132pd_xmm"),
    ("vfmadd132ss xmm0, xmm1, xmm2", "vfmadd132ss"),
    ("vfmadd132sd xmm0, xmm1, xmm2", "vfmadd132sd"),
    ("vfmsub132ps xmm0, xmm1, xmm2", "vfmsub132ps"),
    ("vfmsub213ps xmm0, xmm1, xmm2", "vfmsub213ps"),
    ("vfmsub231ps xmm0, xmm1, xmm2", "vfmsub231ps"),
    ("vfnmadd132ps xmm0, xmm1, xmm2", "vfnmadd132ps"),
    ("vfnmsub231ps xmm0, xmm1, xmm2", "vfnmsub231ps"),
    # AVX2 integer
    ("vpaddb xmm0, xmm1, xmm2", "vpaddb_xmm"),
    ("vpaddb ymm0, ymm1, ymm2", "vpaddb_ymm"),
    ("vpaddw xmm0, xmm1, xmm2", "vpaddw_xmm"),
    ("vpaddd xmm0, xmm1, xmm2", "vpaddd_xmm"),
    ("vpaddq xmm0, xmm1, xmm2", "vpaddq_xmm"),
    ("vpsubb ymm0, ymm1, ymm2", "vpsubb_ymm"),
    ("vpsubw ymm0, ymm1, ymm2", "vpsubw_ymm"),
    ("vpsubd ymm0, ymm1, ymm2", "vpsubd_ymm"),
    ("vpsubq ymm0, ymm1, ymm2", "vpsubq_ymm"),
    ("vpmullw ymm0, ymm1, ymm2", "vpmullw_ymm"),
    ("vpmulld ymm0, ymm1, ymm2", "vpmulld_ymm"),
    ("vpand ymm0, ymm1, ymm2", "vpand_ymm"),
    ("vpor ymm0, ymm1, ymm2", "vpor_ymm"),
    ("vpxor ymm0, ymm1, ymm2", "vpxor_ymm"),
    ("vpandn ymm0, ymm1, ymm2", "vpandn_ymm"),
    ("vpcmpeqb ymm0, ymm1, ymm2", "vpcmpeqb_ymm"),
    ("vpcmpeqd ymm0, ymm1, ymm2", "vpcmpeqd_ymm"),
    ("vpcmpgtb ymm0, ymm1, ymm2", "vpcmpgtb_ymm"),
    ("vpshufb ymm0, ymm1, ymm2", "vpshufb_ymm"),
    ("vpshufd ymm0, ymm1, 0xE4", "vpshufd_ymm"),
    ("vpunpcklbw ymm0, ymm1, ymm2", "vpunpcklbw_ymm"),
    ("vpunpckhbw ymm0, ymm1, ymm2", "vpunpckhbw_ymm"),
    ("vpackuswb ymm0, ymm1, ymm2", "vpackuswb_ymm"),
    ("vpabsb ymm0, ymm1", "vpabsb_ymm"),
    ("vpabsd ymm0, ymm1", "vpabsd_ymm"),
    ("vpermd ymm0, ymm1, ymm2", "vpermd"),
    ("vpermps ymm0, ymm1, ymm2", "vpermps"),
    ("vperm2i128 ymm0, ymm1, ymm2, 0x31", "vperm2i128"),
    ("vinserti128 ymm0, ymm1, xmm2, 1", "vinserti128"),
    ("vextracti128 xmm0, ymm1, 1", "vextracti128"),
    ("vpbroadcastb xmm0, xmm1", "vpbroadcastb_xmm"),
    ("vpbroadcastd ymm0, xmm1", "vpbroadcastd_ymm"),
    ("vpbroadcastq ymm0, xmm1", "vpbroadcastq_ymm"),
    ("vpmovzxbw ymm0, xmm1", "vpmovzxbw_ymm"),
    ("vpmovsxbw ymm0, xmm1", "vpmovsxbw_ymm"),
    ("vpgatherdd xmm0, [rax + xmm1*4], xmm2", "vpgatherdd"),
    ("vmaskmovps xmm0, xmm1, [rax]", "vmaskmovps_load"),
]


@pytest.mark.parametrize("instr,desc", AVX_MISC_TESTS, ids=[d for _, d in AVX_MISC_TESTS])
def test_avx_misc(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — AES-NI
# ===================================================================

AESNI_TESTS = [
    ("aesenc xmm0, xmm1", "aesenc"),
    ("aesenclast xmm0, xmm1", "aesenclast"),
    ("aesdec xmm0, xmm1", "aesdec"),
    ("aesdeclast xmm0, xmm1", "aesdeclast"),
    ("aesimc xmm0, xmm1", "aesimc"),
    ("aeskeygenassist xmm0, xmm1, 0x01", "aeskeygenassist"),
    ("pclmulqdq xmm0, xmm1, 0x00", "pclmulqdq"),
]


@pytest.mark.parametrize("instr,desc", AESNI_TESTS, ids=[d for _, d in AESNI_TESTS])
def test_aesni(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Prefetch, fence, cache
# ===================================================================

CACHE_TESTS = [
    ("lfence", "lfence"),
    ("sfence", "sfence"),
    ("mfence", "mfence"),
    ("prefetcht0 [rax]", "prefetcht0"),
    ("prefetcht1 [rax]", "prefetcht1"),
    ("prefetcht2 [rax]", "prefetcht2"),
    ("prefetchnta [rax]", "prefetchnta"),
    ("clflush [rax]", "clflush"),
    ("pause", "pause"),
    ("rdtsc", "rdtsc"),
    ("rdtscp", "rdtscp"),
    ("cpuid", "cpuid"),
    ("xgetbv", "xgetbv"),
]


@pytest.mark.parametrize("instr,desc", CACHE_TESTS, ids=[d for _, d in CACHE_TESTS])
def test_cache(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — MOVBE, CRC32, POPCNT as single-instruction checks
# ===================================================================

MISC_SINGLE_TESTS = [
    ("movbe rax, [rbx]", "movbe_load"),
    ("movbe [rbx], rax", "movbe_store"),
    ("movbe eax, [rbx]", "movbe_load32"),
    ("crc32 eax, ebx", "crc32_reg"),
    ("popcnt rax, rbx", "popcnt_reg"),
    ("lzcnt rax, rbx", "lzcnt_reg"),
    ("tzcnt rax, rbx", "tzcnt_reg"),
    ("xsave [rax]", "xsave"),
    ("xrstor [rax]", "xrstor"),
]


@pytest.mark.parametrize("instr,desc", MISC_SINGLE_TESTS, ids=[d for _, d in MISC_SINGLE_TESTS])
def test_misc_single(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86_64 — Exact encoding verification (known-good bytes)
# ===================================================================

EXACT_TESTS = [
    ("ret", b"\xc3", "ret"),
    ("nop", b"\x90", "nop"),
    ("syscall", b"\x0f\x05", "syscall"),
    ("int 3", b"\xcc", "int3"),
    ("hlt", b"\xf4", "hlt"),
    ("clc", b"\xf8", "clc"),
    ("stc", b"\xf9", "stc"),
    ("cmc", b"\xf5", "cmc"),
    ("cld", b"\xfc", "cld"),
    ("std", b"\xfd", "std"),
    ("ud2", b"\x0f\x0b", "ud2"),
    ("leave", b"\xc9", "leave"),
    ("int 0x80", b"\xcd\x80", "int_80"),
    ("mov rax, rbx", b"\x48\x89\xd8", "mov_rax_rbx"),
    ("mov eax, ebx", b"\x89\xd8", "mov_eax_ebx"),
    ("push rax", b"\x50", "push_rax"),
    ("push rbx", b"\x53", "push_rbx"),
    ("push rcx", b"\x51", "push_rcx"),
    ("pop rax", b"\x58", "pop_rax"),
    ("pop rbx", b"\x5b", "pop_rbx"),
    ("xor eax, eax", b"\x31\xc0", "xor_eax_eax"),
    ("xor rax, rax", b"\x48\x31\xc0", "xor_rax_rax"),
    ("add al, 1", b"\x04\x01", "add_al_1"),
    ("inc rax", b"\x48\xff\xc0", "inc_rax"),
    ("dec rax", b"\x48\xff\xc8", "dec_rax"),
    ("stosb", b"\xaa", "stosb"),
    ("movsb", b"\xa4", "movsb"),
    ("scasb", b"\xae", "scasb"),
    ("lodsb", b"\xac", "lodsb"),
    ("cmpsb", b"\xa6", "cmpsb"),
    ("rep movsb", b"\xf3\xa4", "rep_movsb"),
    ("rep stosb", b"\xf3\xaa", "rep_stosb"),
    ("repne scasb", b"\xf2\xae", "repne_scasb"),
    ("lahf", b"\x9f", "lahf"),
    ("sahf", b"\x9e", "sahf"),
    ("cbw", b"\x66\x98", "cbw"),
    ("cwde", b"\x98", "cwde"),
    ("cdqe", b"\x48\x98", "cdqe"),
    ("cwd", b"\x66\x99", "cwd"),
    ("cdq", b"\x99", "cdq"),
    ("cqo", b"\x48\x99", "cqo"),
    ("lfence", b"\x0f\xae\xe8", "lfence"),
    ("sfence", b"\x0f\xae\xf8", "sfence"),
    ("mfence", b"\x0f\xae\xf0", "mfence"),
    ("pause", b"\xf3\x90", "pause"),
    ("cpuid", b"\x0f\xa2", "cpuid"),
    ("rdtsc", b"\x0f\x31", "rdtsc"),
]


@pytest.mark.parametrize("instr,expected,desc", EXACT_TESTS, ids=[d for _, _, d in EXACT_TESTS])
def test_exact_encoding(x64, instr, expected, desc):
    assert x64.asm(instr) == expected


# ===================================================================
# x86_64 — Multi-instruction sequences
# ===================================================================

SEQUENCE_TESTS = [
    ("mov rax, rbx; ret", "mov_ret"),
    ("push rax; push rbx; push rcx; pop rcx; pop rbx; pop rax", "push_pop_3"),
    ("xor eax, eax; inc eax; ret", "xor_inc_ret"),
    ("mov rdi, rsi; mov rsi, rdx; mov rdx, rcx; ret", "arg_shuffle"),
    ("test rax, rax\nje done\nnop\ndone:\nret", "test_je_ret"),
]


@pytest.mark.parametrize("instr,desc", SEQUENCE_TESTS, ids=[d for _, d in SEQUENCE_TESTS])
def test_sequences(x64, instr, desc):
    code = x64.asm(instr)
    assert len(code) > 0


# ===================================================================
# x86_64 — Labels
# ===================================================================

LABEL_TESTS = [
    ("loop:\nnop\njmp loop", "backward_jmp"),
    ("jmp end\nnop\nnop\nend:\nret", "forward_jmp"),
    ("1:\nnop\njmp 1b", "numeric_label_back"),
    ("jmp 1f\nnop\n1:\nret", "numeric_label_fwd"),
    ("start:\ncmp rax, 0\nje done\ndec rax\njmp start\ndone:\nret", "loop_pattern"),
]


@pytest.mark.parametrize("instr,desc", LABEL_TESTS, ids=[d for _, d in LABEL_TESTS])
def test_labels(x64, instr, desc):
    assert len(x64.asm(instr)) > 0


# ===================================================================
# x86 32-bit — basic coverage
# ===================================================================

X86_32_TESTS = [
    ("ret", "ret"),
    ("nop", "nop"),
    ("mov eax, ebx", "mov_eax_ebx"),
    ("mov eax, [ebx]", "load"),
    ("mov [ebx], eax", "store"),
    ("push eax", "push_eax"),
    ("pop eax", "pop_eax"),
    ("add eax, ebx", "add"),
    ("sub eax, 42", "sub_imm"),
    ("xor eax, eax", "xor_self"),
    ("int 0x80", "int_80"),
    ("int 3", "int3"),
    ("mul ebx", "mul"),
    ("div ecx", "div"),
    ("lea eax, [ebx + ecx*4 + 8]", "lea"),
    ("rep movsb", "rep_movsb"),
    ("cmpxchg [ebx], ecx", "cmpxchg"),
    ("bswap eax", "bswap"),
    ("movzx eax, bl", "movzx"),
    ("movsx eax, bl", "movsx"),
    ("shld eax, ebx, 4", "shld"),
    ("shrd eax, ebx, cl", "shrd"),
]


@pytest.mark.parametrize("instr,desc", X86_32_TESTS, ids=[d for _, d in X86_32_TESTS])
def test_x86_32(x86, instr, desc):
    assert len(x86.asm(instr)) > 0


# ===================================================================
# Preamble tests
# ===================================================================

class TestPreamble:
    def test_intel_syntax_default(self, x64):
        assert x64.asm("mov rax, rbx") == b"\x48\x89\xd8"

    def test_att_syntax_override(self):
        att = Assembler(triple="x86_64", cpu="", features="", preamble="")
        assert att.asm("movq %rbx, %rax") == b"\x48\x89\xd8"

    def test_custom_preamble(self):
        a = Assembler(triple="x86_64", cpu="", features="",
                      preamble=".intel_syntax noprefix\n.code64")
        assert a.asm("ret") == b"\xc3"


# ===================================================================
# Error handling
# ===================================================================

class TestErrors:
    def test_bad_instruction(self, x64):
        with pytest.raises(AsmError):
            x64.asm("not_an_instruction rax")

    def test_bad_operand(self, x64):
        with pytest.raises(AsmError):
            x64.asm("mov rax, [not_valid_syntax!@#]")

    def test_empty(self, x64):
        assert x64.asm("") == b""
