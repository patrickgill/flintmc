"""Tests for the Disassembler.

Covers: round-trip, multi-arch, error handling, address parameter,
repr, thread safety, edge cases.
"""

import pytest
from flintmc import Assembler, Disassembler, AsmError, DisasmInstruction


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestX86:
    @pytest.fixture(scope="class")
    def asm(self):
        return Assembler.x86_64()

    @pytest.fixture(scope="class")
    def dis(self):
        return Disassembler.x86_64()

    def test_basic(self, dis):
        result = dis(b"\x90")
        assert len(result) == 1
        assert "nop" in result[0].text

    def test_round_trip(self, asm, dis):
        code = asm("mov rax, rbx; nop; ret")
        result = dis(code)
        assert len(result) == 3
        assert "mov" in result[0].text
        assert "nop" in result[1].text
        assert "ret" in result[2].text

    def test_instruction_fields(self, dis):
        result = dis(b"\x90\xc3")
        assert result[0] == DisasmInstruction(offset=0, size=1, code=b"\x90", text="nop")
        assert result[1].offset == 1
        assert result[1].size == 1
        assert result[1].code == b"\xc3"

    def test_intel_syntax(self, dis):
        # Intel syntax uses "mov rax, rbx" not "movq %rbx, %rax"
        code = bytes([0x48, 0x89, 0xd8])
        result = dis(code)
        assert "%" not in result[0].text  # no AT&T register prefix

    def test_address_display(self, dis):
        # jmp with PC-relative should show target address
        result1 = dis(b"\xeb\xfe", address=0)         # jmp -2 from PC=0
        result2 = dis(b"\xeb\xfe", address=0x1000)     # same bytes, different PC
        # Both decode the same instruction but displayed address differs
        assert result1[0].size == result2[0].size == 2

    def test_empty_input(self, dis):
        assert dis(b"") == []

    def test_invalid_bytes(self, dis):
        # 0x06 is not a valid x86_64 instruction (push es, only valid in 32-bit)
        with pytest.raises(AsmError, match="Cannot disassemble"):
            dis(b"\x06")

    def test_multi_byte_instructions(self, dis):
        code = bytes([0x0f, 0x05])  # syscall
        result = dis(code)
        assert len(result) == 1
        assert result[0].size == 2
        assert "syscall" in result[0].text

    def test_frozen_dataclass(self, dis):
        result = dis(b"\x90")
        with pytest.raises(AttributeError):
            result[0].offset = 99


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAArch64:
    @pytest.fixture(scope="class")
    def asm(self):
        return Assembler.aarch64()

    @pytest.fixture(scope="class")
    def dis(self):
        return Disassembler.aarch64()

    def test_round_trip(self, asm, dis):
        code = asm("mov x0, #42; ret")
        result = dis(code)
        assert len(result) == 2
        assert "mov" in result[0].text
        assert "ret" in result[1].text

    def test_fixed_width(self, dis):
        # All AArch64 instructions are 4 bytes
        code = bytes([0x1f, 0x20, 0x03, 0xd5])  # nop
        result = dis(code)
        assert result[0].size == 4


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestARM:
    @pytest.fixture(scope="class")
    def asm(self):
        return Assembler.cortex_m7_dp()

    @pytest.fixture(scope="class")
    def dis(self):
        return Disassembler.cortex_m7_dp()

    def test_round_trip(self, asm, dis):
        code = asm("nop; bx lr")
        result = dis(code)
        assert len(result) == 2
        assert "nop" in result[0].text
        assert "bx" in result[1].text

    def test_thumb_encoding(self, dis):
        # Thumb nop is 2 bytes
        code = bytes([0x00, 0xbf])
        result = dis(code)
        assert result[0].size == 2


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestRepr:
    def test_basic(self):
        dis = Disassembler.x86_64()
        assert "x86_64" in repr(dis)

    def test_with_cpu(self):
        dis = Disassembler.cortex_m7_dp()
        assert "cortex-m7" in repr(dis)


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestCallSyntax:
    def test_callable(self):
        dis = Disassembler.x86_64()
        result = dis(b"\x90")
        assert len(result) == 1

    def test_disasm_method(self):
        dis = Disassembler.x86_64()
        result = dis.disasm(b"\x90")
        assert len(result) == 1


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestConcurrency:
    def test_threaded(self):
        import concurrent.futures
        dis = Disassembler.x86_64()
        code = b"\x90\xc3"

        def worker():
            for _ in range(100):
                result = dis(code)
                assert len(result) == 2
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(worker) for _ in range(4)]
            assert all(f.result() for f in futures)
