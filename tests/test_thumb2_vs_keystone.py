"""Cross-validation: flint-mc output matches keystone for common instructions.

Also demonstrates where keystone fails and flint-mc succeeds.

Requires the `compat` dependency group:
    uv sync --group compat
"""

import pytest

from flintmc import Assembler

ks_available = True
try:
    from keystone import KS_ARCH_ARM, KS_MODE_THUMB, Ks, KsError
except ImportError:
    ks_available = False

pytestmark = pytest.mark.skipif(not ks_available, reason="keystone-engine not installed")


@pytest.fixture(scope="module")
def flint():
    return Assembler.cortex_m7_dp()


@pytest.fixture(scope="module")
def ks():
    return Ks(KS_ARCH_ARM, KS_MODE_THUMB)


def ks_asm(ks_instance: "Ks", source: str) -> bytes:
    """Keystone asm -> bytes helper."""
    encoding, _ = ks_instance.asm(source)
    if encoding is None:
        raise ValueError(f"keystone produced no output for: {source!r}")
    return bytes(encoding)


# ---------------------------------------------------------------------------
# Parity: both assemblers produce identical bytes
# ---------------------------------------------------------------------------

# Non-branch instructions where encoding is unambiguous.
COMMON_INSTRUCTIONS = [
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
    "push {r4, lr}",
    "pop {r4, pc}",
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
    "it eq",
    "dmb sy",
    "dsb sy",
    "isb sy",
    "wfi",
    "wfe",
    "svc #0",
    # VFP basics
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
    "vpush {s0, s1}",
    "vpop {s0, s1}",
]


@pytest.mark.parametrize("instruction", COMMON_INSTRUCTIONS)
def test_parity(flint, ks, instruction):
    """flint-mc and keystone produce identical machine code."""
    flint_bytes = flint.asm(instruction)
    ks_bytes = ks_asm(ks, instruction)
    assert flint_bytes == ks_bytes, (
        f"{instruction!r}:\n"
        f"  flint:    {flint_bytes.hex()}\n"
        f"  keystone: {ks_bytes.hex()}"
    )


# ---------------------------------------------------------------------------
# Branch encoding semantics differ:
# Keystone: "b #0x20" = branch to address 0x20
# LLVM-MC:  "b #0x20" = branch forward 0x20 bytes from current PC
#
# With labels both agree — the literal-offset difference doesn't matter
# for real code.  We test that both produce *valid* encodings here.
# ---------------------------------------------------------------------------

BRANCH_INSTRUCTIONS = [
    "b #0x20",
    "bl #0x100",
    "beq #0x10",
    "cbz r0, #0x8",
    "cbnz r0, #0x8",
]


@pytest.mark.parametrize("instruction", BRANCH_INSTRUCTIONS)
def test_branches_both_assemble(flint, ks, instruction):
    """Both assemblers produce valid branch encodings (semantics differ)."""
    flint_bytes = flint.asm(instruction)
    ks_bytes = ks_asm(ks, instruction)
    assert len(flint_bytes) == len(ks_bytes), (
        f"{instruction!r}: size mismatch "
        f"flint={len(flint_bytes)} vs keystone={len(ks_bytes)}"
    )


# ---------------------------------------------------------------------------
# Keystone gaps: instructions keystone can't assemble at all
# Expected hex verified against llvm-mc output.
# ---------------------------------------------------------------------------

KEYSTONE_GAPS = [
    # M-class special registers
    ("mrs r0, PRIMASK", "eff31080"),
    ("msr PRIMASK, r0", "80f31088"),
    ("mrs r0, BASEPRI", "eff31180"),
    ("msr BASEPRI, r1", "81f31188"),
    ("mrs r0, BASEPRI_MAX", "eff31280"),
    ("mrs r0, FAULTMASK", "eff31380"),
    ("msr FAULTMASK, r0", "80f31388"),
    ("mrs r0, CONTROL", "eff31480"),
    ("msr CONTROL, r0", "80f31488"),
    # FPv5 fused multiply-accumulate
    ("vfma.f32 s0, s1, s2", "a0ee810a"),
    ("vfms.f32 s0, s1, s2", "a0eec10a"),
    ("vfnma.f32 s0, s1, s2", "90eec10a"),
    ("vfnms.f32 s0, s1, s2", "90ee810a"),
    # Integer divide (keystone KS_ERR_ASM in Thumb mode)
    ("udiv r0, r1, r2", "b1fbf2f0"),
    ("sdiv r0, r1, r2", "91fbf2f0"),
]


@pytest.mark.parametrize("instruction,expected_hex", KEYSTONE_GAPS)
def test_keystone_fails(ks, instruction, expected_hex):
    """Confirm keystone cannot assemble these instructions."""
    with pytest.raises((KsError, Exception)):
        ks_asm(ks, instruction)


@pytest.mark.parametrize("instruction,expected_hex", KEYSTONE_GAPS)
def test_flint_handles_gaps(flint, instruction, expected_hex):
    """flint-mc correctly assembles instructions keystone can't."""
    code = flint.asm(instruction)
    assert code == bytes.fromhex(expected_hex), (
        f"{instruction!r}:\n"
        f"  got:      {code.hex()}\n"
        f"  expected: {expected_hex}"
    )
