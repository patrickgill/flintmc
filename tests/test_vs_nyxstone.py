"""Cross-validation: flint-mc output matches nyxstone across architectures.

Both tools are LLVM-backed, so byte output should be identical — this
validates that flint-mc's ELF extraction, preamble handling, and
semicolon splitting are correct.

Nyxstone supported triples:
    x86_32  -> i686-linux-gnu
    x86_64  -> x86_64-linux-gnu
    armv6m  -> armv6m-none-eabi
    armv7m  -> armv7m-none-eabi
    armv8m  -> armv8m.main-none-eabi
    aarch64 -> aarch64-linux-gnueabihf

Requires nyxstone (LLVM 15-18):
    pip install nyxstone
    # or: NYXSTONE_LLVM_PREFIX=/opt/homebrew/opt/llvm pip install nyxstone
"""

import pytest

from flintmc import Assembler

nx_available = True
try:
    from nyxstone import Nyxstone
except ImportError:
    nx_available = False

pytestmark = pytest.mark.skipif(not nx_available, reason="nyxstone not installed")


def nx_asm(nx: "Nyxstone", source: str) -> bytes:
    """Nyxstone assemble -> bytes helper."""
    return bytes(nx.assemble(source))


# ===================================================================
# x86_64
# ===================================================================

@pytest.fixture(scope="module")
def flint_x64():
    return Assembler.x86_64()


@pytest.fixture(scope="module")
def nx_x64():
    return Nyxstone("x86_64")


X64_INSTRUCTIONS = [
    "nop",
    "ret",
    "mov rax, rbx",
    "mov eax, ebx",
    "mov rax, 42",
    "mov eax, 0",
    "add rax, rbx",
    "sub rax, rbx",
    "and rax, rbx",
    "or rax, rbx",
    "xor rax, rbx",
    "cmp rax, rbx",
    "test rax, rbx",
    "add rax, 1",
    "sub rax, 0x100",
    "xor eax, eax",
    "inc rax",
    "dec rax",
    "neg rax",
    "not rax",
    "push rax",
    "push rbx",
    "push r8",
    "pop rax",
    "pop rbx",
    "pop r8",
    "imul rax, rbx",
    "imul rax, rbx, 42",
    "shl rax, 1",
    "shr rax, 4",
    "sar rax, cl",
    "rol rax, 1",
    "ror rax, cl",
    "lea rax, [rbx]",
    "lea rax, [rbx + rcx*4 + 8]",
    "lea rax, [rip + 0x10]",
    "mov rax, [rbx]",
    "mov rax, [rbx + 8]",
    "mov [rbx], rax",
    "mov rax, [rbx + rcx*8]",
    "movzx eax, byte ptr [rbx]",
    "movsx rax, byte ptr [rbx]",
    "movsxd rax, dword ptr [rbx]",
    "bswap rax",
    "bswap eax",
    "xchg rax, rbx",
    "cdqe",
    "cqo",
    "syscall",
    "int 3",
    "ud2",
    "hlt",
    "leave",
    "clc",
    "stc",
    "cld",
    "std",
    # SSE
    "addss xmm0, xmm1",
    "addsd xmm0, xmm1",
    "addps xmm0, xmm1",
    "addpd xmm0, xmm1",
    "mulss xmm0, xmm1",
    "mulsd xmm0, xmm1",
    "movaps xmm0, xmm1",
    "movups xmm0, xmm1",
    "movd eax, xmm0",
    "movq rax, xmm0",
    "cvtsi2ss xmm0, eax",
    "cvtsi2sd xmm0, rax",
    "cvtss2sd xmm0, xmm1",
    "sqrtsd xmm0, xmm1",
    "comiss xmm0, xmm1",
    "ucomisd xmm0, xmm1",
    "pxor xmm0, xmm1",
    "paddd xmm0, xmm1",
    "psubd xmm0, xmm1",
    "pmulld xmm0, xmm1",
    "pshufd xmm0, xmm1, 0xE4",
    "shufps xmm0, xmm1, 0x1B",
    # BMI
    "popcnt rax, rbx",
    "lzcnt rax, rbx",
    "tzcnt rax, rbx",
    "bsf rax, rbx",
    "bsr rax, rbx",
    "bt rax, 7",
    # String ops
    "rep movsb",
    "rep stosb",
    # AVX (VEX prefix)
    "vaddps ymm0, ymm1, ymm2",
    "vaddpd ymm0, ymm1, ymm2",
    "vmovaps ymm0, ymm1",
    "vmulps ymm0, ymm1, ymm2",
    "vxorps ymm0, ymm0, ymm0",
]


@pytest.mark.parametrize("instruction", X64_INSTRUCTIONS)
def test_x64_parity(flint_x64, nx_x64, instruction):
    flint_bytes = flint_x64.asm(instruction)
    nx_bytes = nx_asm(nx_x64, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )


# ===================================================================
# x86_32 (i686)
# ===================================================================

@pytest.fixture(scope="module")
def flint_x86():
    return Assembler.i686()


@pytest.fixture(scope="module")
def nx_x86():
    return Nyxstone("x86_32")


X86_INSTRUCTIONS = [
    "nop",
    "ret",
    "mov eax, ebx",
    "mov eax, 42",
    "mov eax, 0",
    "add eax, ebx",
    "sub eax, ebx",
    "and eax, ebx",
    "or eax, ebx",
    "xor eax, ebx",
    "cmp eax, ebx",
    "test eax, ebx",
    "add eax, 1",
    "sub eax, 0x100",
    "xor eax, eax",
    "inc eax",
    "dec eax",
    "neg eax",
    "not eax",
    "push eax",
    "push ebx",
    "pop eax",
    "pop ebx",
    "imul eax, ebx",
    "imul eax, ebx, 42",
    "shl eax, 1",
    "shr eax, 4",
    "sar eax, cl",
    "rol eax, 1",
    "ror eax, cl",
    "lea eax, [ebx]",
    "lea eax, [ebx + ecx*4 + 8]",
    "mov eax, [ebx]",
    "mov eax, [ebx + 8]",
    "mov [ebx], eax",
    "mov eax, [ebx + ecx*4]",
    "movzx eax, byte ptr [ebx]",
    "movsx eax, byte ptr [ebx]",
    "bswap eax",
    "xchg eax, ebx",
    "cwde",
    "cdq",
    "int 3",
    "int 0x80",
    "ud2",
    "hlt",
    "leave",
    "clc",
    "stc",
    "cld",
    "std",
    # SSE (32-bit mode)
    "addss xmm0, xmm1",
    "addsd xmm0, xmm1",
    "addps xmm0, xmm1",
    "mulss xmm0, xmm1",
    "movaps xmm0, xmm1",
    "pxor xmm0, xmm1",
    "cvtsi2ss xmm0, eax",
    # String ops
    "rep movsb",
    "rep stosd",
    # x87
    "fld dword ptr [eax]",
    "fstp dword ptr [eax]",
    "faddp",
    "fmulp",
]


@pytest.mark.parametrize("instruction", X86_INSTRUCTIONS)
def test_x86_parity(flint_x86, nx_x86, instruction):
    flint_bytes = flint_x86.asm(instruction)
    nx_bytes = nx_asm(nx_x86, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )


# ===================================================================
# AArch64
# ===================================================================

@pytest.fixture(scope="module")
def flint_a64():
    return Assembler.aarch64()


@pytest.fixture(scope="module")
def nx_a64():
    return Nyxstone("aarch64")


A64_INSTRUCTIONS = [
    "nop",
    "ret",
    "mov x0, #42",
    "mov x0, x1",
    "mov w0, w1",
    "add x0, x1, x2",
    "sub x0, x1, x2",
    "mul x0, x1, x2",
    "sdiv x0, x1, x2",
    "udiv x0, x1, x2",
    "and x0, x1, x2",
    "orr x0, x1, x2",
    "eor x0, x1, x2",
    "neg x0, x1",
    "mvn x0, x1",
    "lsl x0, x1, #4",
    "lsr x0, x1, #4",
    "asr x0, x1, #4",
    "clz x0, x1",
    "rbit x0, x1",
    "rev x0, x1",
    "add w0, w1, w2",
    "sub w0, w1, w2",
    "cmp x0, #0",
    "cmp x0, x1",
    "cmn x0, x1",
    "tst x0, #0xff",
    "adds x0, x1, x2",
    "subs x0, x1, x2",
    "csel x0, x1, x2, eq",
    "csinc x0, x1, x2, ne",
    "cset x0, eq",
    "ldr x0, [x1]",
    "ldr x0, [x1, #8]",
    "str x0, [x1]",
    "ldr w0, [x1]",
    "ldrb w0, [x1]",
    "ldrh w0, [x1]",
    "ldrsb x0, [x1]",
    "ldrsh x0, [x1]",
    "ldrsw x0, [x1]",
    "ldr x0, [x1, #16]!",
    "ldr x0, [x1], #16",
    "ldp x0, x1, [sp]",
    "stp x0, x1, [sp, #-16]!",
    "br x0",
    "blr x0",
    "movz x0, #0x1234",
    "movk x0, #0x5678, lsl #16",
    "ubfx x0, x1, #0, #8",
    "bfi x0, x1, #8, #4",
    "svc #0",
    "brk #0",
    "hlt #0",
    "dmb sy",
    "dsb sy",
    "isb",
    "wfi",
    "wfe",
    "mrs x0, NZCV",
    "msr NZCV, x0",
    # FP scalar
    "fmov d0, d1",
    "fmov d0, #1.0",
    "fadd d0, d1, d2",
    "fsub d0, d1, d2",
    "fmul d0, d1, d2",
    "fdiv d0, d1, d2",
    "fsqrt d0, d1",
    "fcmp d0, d1",
    "fcvtzs x0, d0",
    "scvtf d0, x0",
    "fadd s0, s1, s2",
    "fmul s0, s1, s2",
    "fmadd d0, d1, d2, d3",
    # NEON / SIMD
    "add v0.4s, v1.4s, v2.4s",
    "fadd v0.4s, v1.4s, v2.4s",
    "fmul v0.4s, v1.4s, v2.4s",
    "dup v0.4s, v1.s[0]",
    "ins v0.s[0], w0",
    "umov w0, v0.s[0]",
    "movi v0.4s, #0",
    "ldr q0, [x0]",
    "str q0, [x0]",
    # Stack patterns
    "stp x29, x30, [sp, #-16]!",
    # Exclusive
    "ldxr x0, [x1]",
    "stxr w0, x1, [x2]",
]


@pytest.mark.parametrize("instruction", A64_INSTRUCTIONS)
def test_a64_parity(flint_a64, nx_a64, instruction):
    flint_bytes = flint_a64.asm(instruction)
    nx_bytes = nx_asm(nx_a64, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )


# ===================================================================
# ARM Thumb-2 — ARMv7-M (Cortex-M7)
# ===================================================================

@pytest.fixture(scope="module")
def flint_thumb():
    return Assembler.cortex_m7_dp()


@pytest.fixture(scope="module")
def nx_thumb():
    return Nyxstone("armv7m", cpu="cortex-m7", features="+fp-armv8,+fp64")


THUMB_INSTRUCTIONS = [
    "nop",
    "bx lr",
    "mov r0, #0",
    "mov r0, #42",
    "movw r0, #0x1234",
    "movt r0, #0x5678",
    "ldr r0, [r1]",
    "ldr r0, [r1, #4]",
    "str r0, [r1]",
    "str r0, [r1, #8]",
    "add r0, r1, r2",
    "sub r0, r1, #1",
    "cmp r0, #0",
    "and r0, r1, r2",
    "orr r0, r1, r2",
    "eor r0, r1, #0xFF",
    "lsl r0, r1, #2",
    "lsr r0, r1, #4",
    "mul r0, r1, r2",
    "sxtb r0, r1",
    "uxtb r0, r1",
    "sxth r0, r1",
    "uxth r0, r1",
    "rev r0, r1",
    "clz r0, r1",
    "tst r0, r1",
    "dmb sy",
    "dsb sy",
    "isb sy",
    "wfi",
    "wfe",
    "svc #0",
    # VFP
    "vmov s0, r0",
    "vmov r0, s0",
    "vadd.f32 s0, s1, s2",
    "vsub.f32 s0, s1, s2",
    "vmul.f32 s0, s1, s2",
    "vdiv.f32 s0, s1, s2",
    "vcmp.f32 s0, s1",
    "vmrs APSR_nzcv, FPSCR",
    "vldr s0, [r0]",
    "vstr s0, [r0, #4]",
    # M-class special registers (keystone fails on these)
    "mrs r0, PRIMASK",
    "msr PRIMASK, r0",
    "mrs r0, BASEPRI",
    "msr BASEPRI, r1",
    "mrs r0, FAULTMASK",
    "msr CONTROL, r0",
    # FPv5 fused multiply (keystone also fails)
    "vfma.f32 s0, s1, s2",
    "vfms.f32 s0, s1, s2",
    # Integer divide
    "udiv r0, r1, r2",
    "sdiv r0, r1, r2",
]


@pytest.mark.parametrize("instruction", THUMB_INSTRUCTIONS)
def test_thumb_parity(flint_thumb, nx_thumb, instruction):
    flint_bytes = flint_thumb.asm(instruction)
    nx_bytes = nx_asm(nx_thumb, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )


# ===================================================================
# ARM Thumb-2 — ARMv6-M (Cortex-M0)
# ===================================================================

@pytest.fixture(scope="module")
def flint_m0():
    return Assembler.cortex_m0()


@pytest.fixture(scope="module")
def nx_m0():
    return Nyxstone("armv6m")


M0_INSTRUCTIONS = [
    "nop",
    "bx lr",
    "mov r0, #0",
    "mov r0, #42",
    "mov r0, r1",
    "ldr r0, [r1]",
    "ldr r0, [r1, #4]",
    "str r0, [r1]",
    "str r0, [r1, #4]",
    "add r0, r1, r2",
    "add r0, r0, #1",
    "sub r0, r1, r2",
    "sub r0, r0, #1",
    "cmp r0, #0",
    "cmp r0, r1",
    "and r0, r1",
    "orr r0, r1",
    "eor r0, r1",
    "lsl r0, r1, #2",
    "lsr r0, r1, #4",
    "asr r0, r1, #4",
    "mul r0, r1, r0",
    "mvn r0, r1",
    "neg r0, r1",
    "tst r0, r1",
    "sxtb r0, r1",
    "uxtb r0, r1",
    "sxth r0, r1",
    "uxth r0, r1",
    "rev r0, r1",
    "rev16 r0, r1",
    "push {r0, lr}",
    "pop {r0, pc}",
    "ldrb r0, [r1]",
    "strb r0, [r1]",
    "ldrh r0, [r1]",
    "strh r0, [r1]",
    "ldrsb r0, [r1, r2]",
    "ldrsh r0, [r1, r2]",
    "svc #0",
    "wfi",
    "wfe",
    "dmb sy",
    "dsb sy",
    "isb sy",
]


@pytest.mark.parametrize("instruction", M0_INSTRUCTIONS)
def test_m0_parity(flint_m0, nx_m0, instruction):
    flint_bytes = flint_m0.asm(instruction)
    nx_bytes = nx_asm(nx_m0, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )


# ===================================================================
# ARM Thumb-2 — ARMv8-M (Cortex-M33)
# ===================================================================

@pytest.fixture(scope="module")
def flint_m33():
    return Assembler.cortex_m33()


@pytest.fixture(scope="module")
def nx_m33():
    return Nyxstone("armv8m", cpu="cortex-m33")


M33_INSTRUCTIONS = [
    "nop",
    "bx lr",
    "mov r0, #0",
    "mov r0, #42",
    "movw r0, #0x1234",
    "movt r0, #0x5678",
    "ldr r0, [r1]",
    "str r0, [r1]",
    "add r0, r1, r2",
    "sub r0, r1, #1",
    "cmp r0, #0",
    "and r0, r1, r2",
    "orr r0, r1, r2",
    "eor r0, r1, #0xFF",
    "udiv r0, r1, r2",
    "sdiv r0, r1, r2",
    "clz r0, r1",
    "sxtb r0, r1",
    "uxtb r0, r1",
    "tst r0, r1",
    "dmb sy",
    "dsb sy",
    "isb sy",
    "wfi",
    "wfe",
    "svc #0",
    # M-class special registers
    "mrs r0, PRIMASK",
    "msr PRIMASK, r0",
    "mrs r0, BASEPRI",
    "mrs r0, FAULTMASK",
    "msr CONTROL, r0",
    # TrustZone
    "sg",
]


@pytest.mark.parametrize("instruction", M33_INSTRUCTIONS)
def test_m33_parity(flint_m33, nx_m33, instruction):
    flint_bytes = flint_m33.asm(instruction)
    nx_bytes = nx_asm(nx_m33, instruction)
    assert flint_bytes == nx_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  nyxstone: {nx_bytes.hex()}"
    )
