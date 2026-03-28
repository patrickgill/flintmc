"""Tests for AVR assembly.

Covers basic arithmetic, logic, and register access for Atmel AVR.
"""

import pytest
from flintmc import Assembler, AsmError

@pytest.fixture(scope="module")
def avr():
    return Assembler.avr()

def test_avr_basic(avr):
    # add rd, rr -> 0000 11rd dddd rrrr (16-bit, little-endian in ELF)
    # r16=10000, r17=10001 -> 0x0f01 stored as [0x01, 0x0f]
    code = avr("add r16, r17")
    assert code == b"\x01\x0f"

def test_avr_arithmetic(avr):
    code = avr("""
        add r0, r1
        sub r2, r3
        adc r4, r5
        and r6, r7
        or  r8, r9
        eor r10, r11
    """)
    assert len(code) == 12

def test_avr_immediate(avr):
    # ldi rd, k -> 1110 kkkk dddd kkkk
    # rd must be r16-r31
    code = avr("ldi r16, 0x42")
    assert len(code) == 2

def test_avr_error(avr):
    with pytest.raises(AsmError):
        avr("ldi r0, 0x42")  # r0 cannot be used with ldi
