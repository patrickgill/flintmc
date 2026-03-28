"""Tests for LoongArch64 assembly.

Covers basic arithmetic and logic for LoongArch.
"""

import pytest
from flintmc import Assembler, AsmError

@pytest.fixture(scope="module")
def la64():
    return Assembler.loongarch64()

def test_la64_basic(la64):
    # add.d $r4, $r5, $r6
    code = la64("add.d $r4, $r5, $r6")
    assert len(code) == 4

def test_la64_arithmetic(la64):
    code = la64("""
        sub.d $r1, $r2, $r3
        and $r10, $r11, $r12
        or $r20, $r21, $r22
        xor $r28, $r29, $r30
    """)
    assert len(code) == 16

def test_la64_error(la64):
    with pytest.raises(AsmError):
        la64("invalid_instr $r1, $r2")
