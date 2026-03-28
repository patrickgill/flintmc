"""Tests for MSP430 assembly.

Covers basic arithmetic and logic for TI MSP430.
"""

import pytest
from flintmc import Assembler, AsmError

@pytest.fixture(scope="module")
def msp430():
    return Assembler.msp430()

def test_msp430_basic(msp430):
    # add r1, r2
    code = msp430("add r1, r2")
    assert len(code) == 2

def test_msp430_arithmetic(msp430):
    code = msp430("""
        add r1, r2
        sub r3, r4
        xor r5, r6
        and r7, r8
    """)
    assert len(code) == 8

def test_msp430_error(msp430):
    with pytest.raises(AsmError):
        msp430("invalid_instruction r1, r2")
