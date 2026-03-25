"""Cross-validation: flint-mc output matches arm-none-eabi-as (GNU assembler).

GAS is the ground-truth reference assembler for ARM — what silicon vendors
validate against and what every firmware gets built with.  If flint-mc and
GAS agree on an encoding, it's correct.

Skipped automatically if arm-none-eabi-as is not installed.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from flintmc import Assembler

GAS = shutil.which("arm-none-eabi-as")
OBJCOPY = shutil.which("arm-none-eabi-objcopy")

pytestmark = pytest.mark.skipif(
    not GAS or not OBJCOPY,
    reason="arm-none-eabi toolchain not installed",
)


@pytest.fixture(scope="module")
def flint():
    return Assembler()


def gas_asm(source: str, cpu: str = "cortex-m7", fpu: str = "fpv5-d16") -> bytes:
    """Assemble via arm-none-eabi-as and extract .text bytes."""
    full = f".syntax unified\n.thumb\n{source}\n"
    with tempfile.NamedTemporaryFile(suffix=".o") as obj, \
         tempfile.NamedTemporaryFile(suffix=".bin") as out:
        result = subprocess.run(
            [GAS, f"-mcpu={cpu}", f"-mfpu={fpu}", "-mthumb", "-o", obj.name, "-"],
            input=full.encode(),
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        subprocess.run(
            [OBJCOPY, "-O", "binary", "-j", ".text", obj.name, out.name],
            check=True, capture_output=True,
        )
        return Path(out.name).read_bytes()


# ---------------------------------------------------------------------------
# Comprehensive instruction corpus — verified against GAS ground truth
# ---------------------------------------------------------------------------

# (mnemonic, description) pairs
# Every instruction here is something we actually use or expect to use
# in Cortex-M7 firmware emulation.

INSTRUCTIONS = [
    # --- Basic ALU ---
    ("nop", "no-op"),
    ("bx lr", "return"),
    ("mov r0, #0", "clear register"),
    ("mov r0, #42", "small immediate"),
    ("movw r0, #0x1234", "16-bit immediate"),
    ("movt r0, #0x5678", "upper 16-bit immediate"),
    ("add r0, r1, r2", "register add"),
    ("adds r0, r1, #1", "add with flags"),
    ("sub r0, r1, #1", "subtract immediate"),
    ("rsb r0, r1, #0", "reverse subtract (negate)"),
    ("cmp r0, #0", "compare immediate"),
    ("cmn r0, r1", "compare negative"),
    ("and r0, r1, r2", "bitwise AND"),
    ("orr r0, r1, r2", "bitwise OR"),
    ("eor r0, r1, #0xFF", "bitwise XOR immediate"),
    ("bic r0, r1, #0xF0", "bit clear"),
    ("mvn r0, r1", "bitwise NOT"),
    ("tst r0, r1", "test bits"),
    ("lsl r0, r1, #2", "logical shift left"),
    ("lsr r0, r1, #4", "logical shift right"),
    ("asr r0, r1, #8", "arithmetic shift right"),
    ("ror r0, r1, r2", "rotate right"),
    ("mul r0, r1, r2", "multiply"),
    ("mla r0, r1, r2, r3", "multiply-accumulate"),
    ("mls r0, r1, r2, r3", "multiply-subtract"),
    ("udiv r0, r1, r2", "unsigned divide"),
    ("sdiv r0, r1, r2", "signed divide"),
    ("smull r0, r1, r2, r3", "signed 64-bit multiply"),
    ("umull r0, r1, r2, r3", "unsigned 64-bit multiply"),
    ("clz r0, r1", "count leading zeros"),
    ("rbit r0, r1", "reverse bits"),
    ("rev r0, r1", "reverse bytes"),
    ("rev16 r0, r1", "reverse bytes in halfwords"),
    ("revsh r0, r1", "reverse bytes signed halfword"),
    ("sxtb r0, r1", "sign-extend byte"),
    ("uxtb r0, r1", "zero-extend byte"),
    ("sxth r0, r1", "sign-extend halfword"),
    ("uxth r0, r1", "zero-extend halfword"),

    # --- Load/store ---
    ("ldr r0, [r1]", "load word"),
    ("ldr r0, [r1, #4]", "load word offset"),
    ("ldr r0, [r1], #4", "load word post-increment"),
    ("ldr r0, [r1, #4]!", "load word pre-increment"),
    ("str r0, [r1]", "store word"),
    ("str r0, [r1, #8]", "store word offset"),
    ("ldrb r0, [r1]", "load byte"),
    ("ldrh r0, [r1]", "load halfword"),
    ("ldrsb r0, [r1, r2]", "load signed byte"),
    ("ldrsh r0, [r1, r2]", "load signed halfword"),
    ("strb r0, [r1]", "store byte"),
    ("strh r0, [r1]", "store halfword"),
    ("ldr.w r0, [r1, r2, lsl #2]", "load shifted register"),
    ("ldm r0, {r1, r2, r3}", "load multiple"),
    ("stm r0!, {r1, r2, r3}", "store multiple writeback"),
    ("push {r4, r5, lr}", "push"),
    ("pop {r4, r5, pc}", "pop"),

    # --- M-class special registers (keystone gaps) ---
    ("mrs r0, PRIMASK", "read PRIMASK"),
    ("msr PRIMASK, r0", "write PRIMASK"),
    ("mrs r0, BASEPRI", "read BASEPRI"),
    ("msr BASEPRI, r1", "write BASEPRI"),
    ("mrs r0, BASEPRI_MAX", "read BASEPRI_MAX"),
    ("mrs r0, FAULTMASK", "read FAULTMASK"),
    ("msr FAULTMASK, r0", "write FAULTMASK"),
    ("mrs r0, CONTROL", "read CONTROL"),
    ("msr CONTROL, r0", "write CONTROL"),
    ("mrs r0, MSP", "read MSP"),
    ("msr MSP, r0", "write MSP"),
    ("mrs r0, PSP", "read PSP"),
    ("msr PSP, r0", "write PSP"),

    # --- IT blocks ---
    ("it eq\nmoveq r0, #1", "IT EQ"),
    ("ite ne\nmovne r0, #1\nmoveq r0, #0", "ITE NE"),

    # --- System ---
    ("cpsid i", "disable interrupts"),
    ("cpsie i", "enable interrupts"),
    ("dmb sy", "data memory barrier"),
    ("dsb sy", "data synchronization barrier"),
    ("isb sy", "instruction synchronization barrier"),
    ("wfi", "wait for interrupt"),
    ("wfe", "wait for event"),
    ("sev", "send event"),
    ("svc #0", "supervisor call"),
    ("bkpt #0", "breakpoint"),

    # --- FPU basic ---
    ("vmov s0, r0", "ARM to FP register"),
    ("vmov r0, s0", "FP to ARM register"),
    ("vadd.f32 s0, s1, s2", "FP add"),
    ("vsub.f32 s0, s1, s2", "FP subtract"),
    ("vmul.f32 s0, s1, s2", "FP multiply"),
    ("vdiv.f32 s0, s1, s2", "FP divide"),
    ("vneg.f32 s0, s1", "FP negate"),
    ("vabs.f32 s0, s1", "FP absolute"),
    ("vsqrt.f32 s0, s1", "FP square root"),
    ("vcmp.f32 s0, s1", "FP compare"),
    ("vcmp.f32 s0, #0.0", "FP compare zero"),
    ("vmrs APSR_nzcv, FPSCR", "FP flags to APSR"),
    ("vcvt.f32.s32 s0, s0", "int to float"),
    ("vcvt.s32.f32 s0, s0", "float to int"),
    ("vcvt.f64.s32 d0, s0", "int to double"),
    ("vcvt.f32.f64 s0, d0", "double to float"),
    ("vldr s0, [r0]", "FP load"),
    ("vstr s0, [r0, #4]", "FP store"),
    ("vpush {s0, s1}", "FP push"),
    ("vpop {s0, s1}", "FP pop"),
    ("vmov s31, r0", "high FP register"),
    ("vmov.f64 d15, d0", "double register"),

    # --- FPv5 fused ops (keystone gaps) ---
    ("vfma.f32 s0, s1, s2", "fused multiply-add"),
    ("vfms.f32 s0, s1, s2", "fused multiply-sub"),
    ("vfnma.f32 s0, s1, s2", "fused neg multiply-add"),
    ("vfnms.f32 s0, s1, s2", "fused neg multiply-sub"),

    # --- DSP / saturating ---
    ("usat r3, #15, r3", "unsigned saturate"),
    ("ssat r0, #16, r1", "signed saturate"),
    ("qadd r0, r1, r2", "saturating add"),
    ("qsub r0, r1, r2", "saturating subtract"),
    ("smmul r0, r1, r2", "signed most-significant multiply"),
    ("smmla r0, r1, r2, r3", "signed MSW multiply-accumulate"),
    ("pkhbt r0, r1, r2", "pack halfword bottom-top"),
    ("pkhtb r0, r1, r2, asr #16", "pack halfword top-bottom"),

    # --- Width specifiers ---
    ("nop.n", "narrow NOP"),
    ("nop.w", "wide NOP"),

    # --- Thumb-2 modified immediates ---
    ("mov.w r0, #0x01010101", "repeated byte pattern"),
    ("orr r0, r0, #0x10001", "shifted immediate"),
]


@pytest.mark.parametrize(
    "instruction,desc",
    INSTRUCTIONS,
    ids=[f"{desc}" for _, desc in INSTRUCTIONS],
)
def test_matches_gas(flint, instruction, desc):
    """flint-mc produces identical bytes to arm-none-eabi-as."""
    flint_bytes = flint.asm(instruction)
    gas_bytes = gas_asm(instruction)
    assert flint_bytes == gas_bytes, (
        f"{desc}: {instruction!r}\n"
        f"  flint: {flint_bytes.hex()}\n"
        f"  GAS:   {gas_bytes.hex()}"
    )
