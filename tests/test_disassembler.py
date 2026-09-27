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
        assert result[0] == DisasmInstruction(offset=0, size=1, code=b"\x90", text="nop", address=0)
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

    def test_invalid_bytes_strict(self, dis):
        # 0x06 is not a valid x86_64 instruction (push es, only valid in 32-bit)
        with pytest.raises(AsmError, match="Cannot disassemble"):
            dis(b"\x06", strict=True)

    def test_invalid_bytes_skipped(self, dis):
        """Default non-strict mode skips bad bytes."""
        assert dis(b"\x06") == []

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

    @pytest.mark.parametrize("triple", ["aarch64", "arm64-apple-macos"])
    def test_skip_invalid_stays_aligned(self, triple):
        # arm64 contains "arm" — must still skip 4 bytes, not 2
        dis = Disassembler(triple)
        result = dis(b"\xff\xff\xff\xff" + bytes([0x1f, 0x20, 0x03, 0xd5]))
        assert [(i.offset, i.text) for i in result] == [(4, "nop")]
        assert dis._min_insn_size == 4


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
class TestCapstoneCompat:
    """Adversarial tests for address, mnemonic, and op_str attributes."""

    @pytest.fixture(scope="class")
    def dis(self):
        return Disassembler.x86_64()

    @pytest.fixture(scope="class")
    def arm_dis(self):
        return Disassembler.cortex_m7_dp()

    # --- address ---

    def test_address_zero_base(self, dis):
        """address defaults to offset when base is 0."""
        result = dis(b"\x90\xc3", address=0)
        assert result[0].address == 0
        assert result[1].address == 1

    def test_address_nonzero_base(self, dis):
        """address = base + offset for each instruction."""
        result = dis(b"\x90\xc3", address=0xDEAD_0000)
        assert result[0].address == 0xDEAD_0000
        assert result[1].address == 0xDEAD_0001

    def test_address_large_base(self, dis):
        """64-bit addresses don't truncate."""
        base = 0xFFFF_FFFF_FFFF_0000
        result = dis(b"\x90", address=base)
        assert result[0].address == base

    def test_address_multi_byte_offsets(self, dis):
        """address tracks correctly through variable-length instructions."""
        # mov eax, 1 (5 bytes) + nop (1 byte) + ret (1 byte)
        code = b"\xb8\x01\x00\x00\x00\x90\xc3"
        result = dis(code, address=0x1000)
        assert result[0].address == 0x1000
        assert result[1].address == 0x1005
        assert result[2].address == 0x1006

    def test_address_default_base(self, dis):
        """Default base address is 0."""
        result = dis(b"\x90")
        assert result[0].address == 0

    def test_address_thumb_mixed_sizes(self, arm_dis):
        """Thumb: 2-byte and 4-byte instructions get correct addresses."""
        # nop (2B) + nop (2B)
        code = bytes([0x00, 0xBF, 0x00, 0xBF])
        result = arm_dis(code, address=0x0800_0000)
        assert result[0].address == 0x0800_0000
        assert result[1].address == 0x0800_0002

    def test_address_is_frozen(self, dis):
        """address field is immutable (frozen dataclass)."""
        result = dis(b"\x90")
        with pytest.raises(AttributeError):
            result[0].address = 0x9999

    # --- mnemonic ---

    def test_mnemonic_no_operands(self, dis):
        """Instruction with no operands: mnemonic is the full text."""
        result = dis(b"\x90")
        assert result[0].mnemonic == "nop"

    def test_mnemonic_with_operands(self, dis):
        """Instruction with operands: mnemonic is first token only."""
        code = b"\xb8\x01\x00\x00\x00"  # mov eax, 1
        result = dis(code)
        assert result[0].mnemonic == "mov"

    def test_mnemonic_prefix_instruction(self, dis):
        """rep-prefixed instruction mnemonic."""
        code = b"\xf3\xa4"  # rep movsb
        result = dis(code)
        assert "rep" in result[0].mnemonic or "movs" in result[0].mnemonic

    def test_mnemonic_arm(self, arm_dis):
        """ARM mnemonic extraction works."""
        code = bytes([0x70, 0x47])  # bx lr
        result = arm_dis(code)
        assert result[0].mnemonic == "bx"

    # --- op_str ---

    def test_op_str_no_operands(self, dis):
        """No-operand instruction has empty op_str."""
        result = dis(b"\x90")
        assert result[0].op_str == ""

    def test_op_str_single_operand(self, dis):
        """Single operand instruction."""
        code = b"\xff\xd0"  # call rax
        result = dis(code)
        assert result[0].op_str != ""
        assert result[0].mnemonic == "call"

    def test_op_str_multiple_operands(self, dis):
        """Multi-operand: op_str contains comma-separated operands."""
        code = b"\xb8\x01\x00\x00\x00"  # mov eax, 1
        result = dis(code)
        assert "," in result[0].op_str
        assert "eax" in result[0].op_str

    def test_op_str_memory_operand(self, dis):
        """Memory operands with brackets preserved."""
        code = b"\x8b\x00"  # mov eax, dword ptr [rax]
        result = dis(code)
        assert "[" in result[0].op_str

    def test_op_str_arm_operands(self, arm_dis):
        """ARM operand string extraction."""
        code = bytes([0x70, 0x47])  # bx lr
        result = arm_dis(code)
        assert "lr" in result[0].op_str

    # --- mnemonic + op_str reconstruct text ---

    def test_mnemonic_op_str_cover_text(self, dis):
        """mnemonic + op_str account for all content in text."""
        code = b"\xb8\x01\x00\x00\x00\x90\xc3"
        for insn in dis(code, address=0x1000):
            # text may use tab between mnemonic and operands; normalize
            assert insn.mnemonic in insn.text
            if insn.op_str:
                assert insn.op_str in insn.text

    # --- equality with address ---

    def test_equality_different_address(self, dis):
        """Same bytes at different addresses are not equal."""
        r1 = dis(b"\x90", address=0x1000)
        r2 = dis(b"\x90", address=0x2000)
        assert r1[0] != r2[0]

    def test_equality_same_address(self, dis):
        """Same bytes at same address are equal."""
        r1 = dis(b"\x90", address=0x1000)
        r2 = dis(b"\x90", address=0x1000)
        assert r1[0] == r2[0]

    # --- .bytes alias ---

    def test_bytes_alias(self, dis):
        """.bytes returns same object as .code."""
        result = dis(b"\x90")
        assert result[0].bytes is result[0].code
        assert result[0].bytes == b"\x90"

    def test_bytes_multi_byte(self, dis):
        """.bytes works for multi-byte instructions."""
        code = b"\xb8\x01\x00\x00\x00"  # mov eax, 1
        result = dis(code)
        assert result[0].bytes == code

    # --- empty input ---

    def test_empty_gives_no_addresses(self, dis):
        """Empty input produces no instructions, no addresses to check."""
        assert dis(b"", address=0xBAAD) == []


@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestCountAndStrict:
    """Adversarial tests for count and strict parameters."""

    @pytest.fixture(scope="class")
    def dis(self):
        return Disassembler.x86_64()

    @pytest.fixture(scope="class")
    def arm_dis(self):
        return Disassembler.cortex_m7_dp()

    # --- count ---

    def test_count_zero_means_all(self, dis):
        """count=0 decodes everything (default)."""
        code = b"\x90\x90\x90\xc3"
        assert len(dis(code, count=0)) == 4

    def test_count_limits_instructions(self, dis):
        """count=2 stops after 2 instructions."""
        code = b"\x90\x90\x90\xc3"
        result = dis(code, count=2)
        assert len(result) == 2
        assert result[-1].mnemonic == "nop"

    def test_count_one(self, dis):
        """count=1 returns exactly one instruction."""
        code = b"\x90\xc3"
        result = dis(code, count=1)
        assert len(result) == 1

    def test_count_exceeds_instructions(self, dis):
        """count larger than available instructions returns all."""
        code = b"\x90\xc3"
        result = dis(code, count=100)
        assert len(result) == 2

    def test_count_with_address(self, dis):
        """count and address combine correctly."""
        code = b"\x90\x90\x90"
        result = dis(code, address=0x1000, count=2)
        assert len(result) == 2
        assert result[0].address == 0x1000
        assert result[1].address == 0x1001

    def test_count_on_variable_length(self, dis):
        """count works with variable-length instructions."""
        # mov eax, 1 (5B) + nop (1B) + ret (1B)
        code = b"\xb8\x01\x00\x00\x00\x90\xc3"
        result = dis(code, count=1)
        assert len(result) == 1
        assert result[0].size == 5

    def test_count_empty_input(self, dis):
        """count with empty input returns empty."""
        assert dis(b"", count=5) == []

    # --- strict=False (skip bad bytes) ---

    def test_nonstrict_skips_bad_byte_x86(self, dis):
        """Non-strict skips bad bytes and continues decoding."""
        # 0x06 is invalid in x86_64, but nop after it should decode
        code = b"\x90\x06\x90"
        result = dis(code, strict=False)
        assert len(result) == 2
        assert all(r.mnemonic == "nop" for r in result)

    def test_nonstrict_preserves_addresses(self, dis):
        """Addresses stay correct after skipping bad bytes."""
        code = b"\x90\x06\xc3"  # nop, bad, ret
        result = dis(code, address=0x1000, strict=False)
        assert result[0].address == 0x1000
        assert result[1].address == 0x1002  # skipped 0x1001

    def test_nonstrict_all_bad_returns_empty(self, dis):
        """All-bad input in non-strict mode returns empty list."""
        result = dis(b"\x06\x06\x06", strict=False)
        assert result == []

    def test_nonstrict_trailing_bad_bytes(self, dis):
        """Bad bytes at end don't affect decoded instructions."""
        code = b"\x90\xc3\x06\x06"
        result = dis(code, strict=False)
        assert len(result) == 2

    def test_nonstrict_bad_bytes_at_start(self, dis):
        """Bad bytes at start are skipped, rest decodes."""
        code = b"\x06\x06\x90"
        result = dis(code, strict=False)
        assert len(result) == 1
        assert result[0].mnemonic == "nop"

    def test_nonstrict_thumb_skips_by_two(self, arm_dis):
        """Thumb non-strict skips 2 bytes (halfword-aligned)."""
        # 0x00 0x00 is udf #0 in Thumb — valid but let's use known-bad bytes
        # Build: valid nop + 2 bad bytes + valid nop
        nop = bytes([0x00, 0xBF])
        bad = bytes([0x99, 0x99])  # not a valid Thumb instruction
        code = nop + bad + nop
        result = arm_dis(code, address=0x0800_0000, strict=False)
        # Should get at least the two nops; bad pair skipped as one unit
        nop_results = [r for r in result if r.mnemonic == "nop"]
        assert len(nop_results) == 2
        assert nop_results[0].address == 0x0800_0000
        assert nop_results[1].address == 0x0800_0004

    def test_strict_raises(self, dis):
        """Explicit strict=True raises on bad bytes."""
        with pytest.raises(AsmError, match="Cannot disassemble"):
            dis(b"\x06", strict=True)

    # --- count + strict combined ---

    def test_count_with_nonstrict(self, dis):
        """count limits even when skipping bad bytes."""
        code = b"\x90\x06\x90\xc3"
        result = dis(code, count=1, strict=False)
        assert len(result) == 1

    def test_count_with_nonstrict_skips_then_limits(self, dis):
        """Bad bytes don't count toward the limit."""
        code = b"\x06\x90\x90\xc3"  # bad, nop, nop, ret
        result = dis(code, count=2, strict=False)
        assert len(result) == 2

    # --- callable syntax ---

    def test_callable_with_count(self, dis):
        """__call__ forwards count."""
        assert len(dis(b"\x90\x90\xc3", count=1)) == 1

    def test_callable_with_strict(self, dis):
        """__call__ forwards strict."""
        result = dis(b"\x90\x06\xc3", strict=False)
        assert len(result) == 2


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
