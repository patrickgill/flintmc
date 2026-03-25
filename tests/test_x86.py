"""Tests for x86 and x86_64 assembly."""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def x64():
    return Assembler.x86_64()


@pytest.fixture(scope="module")
def x86():
    return Assembler.i686()


# ---------------------------------------------------------------------------
# x86_64 basics
# ---------------------------------------------------------------------------

class TestX86_64:
    def test_ret(self, x64):
        assert x64.asm("ret") == b"\xc3"

    def test_nop(self, x64):
        assert x64.asm("nop") == b"\x90"

    def test_mov_reg_reg(self, x64):
        code = x64.asm("mov rax, rbx")
        assert code == b"\x48\x89\xd8"

    def test_mov_reg_imm(self, x64):
        code = x64.asm("mov rax, 0x42")
        assert len(code) > 0

    def test_multi(self, x64):
        code = x64.asm("mov rax, rbx; ret")
        assert code == b"\x48\x89\xd8\xc3"

    def test_push_pop(self, x64):
        code = x64.asm("push rax; pop rbx")
        assert len(code) > 0

    def test_syscall(self, x64):
        code = x64.asm("syscall")
        assert code == b"\x0f\x05"

    def test_labels(self, x64):
        code = x64.asm("loop:\nnop\njmp loop")
        assert len(code) > 0

    def test_lea(self, x64):
        code = x64.asm("lea rax, [rbx + rcx*4 + 8]")
        assert len(code) > 0

    def test_sse(self, x64):
        code = x64.asm("movaps xmm0, xmm1")
        assert len(code) > 0

    def test_avx(self, x64):
        code = x64.asm("vaddps ymm0, ymm1, ymm2")
        assert len(code) > 0

    def test_error(self, x64):
        with pytest.raises(AsmError):
            x64.asm("not_an_instruction rax")


# ---------------------------------------------------------------------------
# x86 32-bit
# ---------------------------------------------------------------------------

class TestX86_32:
    def test_ret(self, x86):
        assert x86.asm("ret") == b"\xc3"

    def test_mov_reg_reg(self, x86):
        code = x86.asm("mov eax, ebx")
        assert code == b"\x89\xd8"

    def test_int(self, x86):
        code = x86.asm("int 0x80")
        assert code == b"\xcd\x80"


# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------

class TestPreamble:
    def test_intel_syntax_default(self, x64):
        """x86_64 profile defaults to Intel syntax."""
        # Intel: mov rax, rbx
        # AT&T:  movq %rbx, %rax
        assert x64.asm("mov rax, rbx") == b"\x48\x89\xd8"

    def test_att_syntax_override(self):
        """Override preamble to AT&T syntax."""
        att = Assembler(triple="x86_64", cpu="", features="", preamble="")
        code = att.asm("movq %rbx, %rax")
        assert code == b"\x48\x89\xd8"

    def test_custom_preamble(self):
        """Pass arbitrary preamble."""
        a = Assembler(triple="x86_64", cpu="", features="",
                      preamble=".intel_syntax noprefix\n.code64")
        assert a.asm("ret") == b"\xc3"
