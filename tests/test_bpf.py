"""Tests for BPF assembly.

Covers basic arithmetic and logic for eBPF.
"""

import pytest
from flintmc import Assembler, AsmError

@pytest.fixture(scope="module")
def bpf():
    return Assembler.bpf()

def test_bpf_basic(bpf):
    # r1 = 1
    code = bpf("r1 = 1")
    assert len(code) == 8
    # 0xb7 0x01 0x00 0x00 0x01 0x00 0x00 0x00
    assert code[0] == 0xb7
    assert code[1] == 0x01

def test_bpf_arithmetic(bpf):
    code = bpf("""
        r1 += r2
        r3 -= r4
        r5 *= r6
        r7 /= 8
    """)
    assert len(code) == 32

def test_bpf_error(bpf):
    with pytest.raises(AsmError):
        bpf("mov r1, r2")  # Invalid syntax for BPF in LLVM-MC (usually)
