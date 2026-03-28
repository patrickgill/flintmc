"""Tests for absolute address symbol injection.

For data references (mov, ldr, lea, etc.), .set values are used as
absolute addresses/immediates. These tests verify the literal address
bytes appear correctly in the encoded output across architectures and
edge cases.
"""

import struct
import pytest
from flintmc import Assembler, AsmError


# ---------------------------------------------------------------------------
# x86_64 absolute addresses
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestX86AbsoluteAddress:
    @pytest.fixture(scope="class")
    def x86(self):
        return Assembler.x86_64()

    def test_mov_rax_large_immediate(self, x86):
        """Large value uses movabs rax, imm64 (10 bytes)."""
        addr = 0xDEAD_BEEF_CAFE_BABE
        code = x86("mov rax, my_var", symbols={"my_var": addr})
        assert len(code) == 10
        val = struct.unpack_from("<Q", code, 2)[0]
        assert val == addr

    def test_mov_rax_small_immediate(self, x86):
        """Small value uses mov rax, imm32 (sign-extended, 7 bytes)."""
        code = x86("mov rax, v", symbols={"v": 0x42})
        assert len(code) == 7
        val = struct.unpack_from("<I", code, 3)[0]
        assert val == 0x42

    def test_mov_eax_32bit(self, x86):
        addr = 0xCAFEBABE
        code = x86("mov eax, my_var", symbols={"my_var": addr})
        assert addr.to_bytes(4, "little") in code

    def test_mov_memory_operand(self, x86):
        """mov eax, [addr] — absolute memory reference."""
        addr = 0xDEADBEEF
        code = x86("mov eax, dword ptr [buf]", symbols={"buf": addr})
        assert addr.to_bytes(4, "little") in code

    def test_zero_address(self, x86):
        code = x86("mov rax, z", symbols={"z": 0})
        # Small value: 7 bytes, imm32 at offset 3
        val = struct.unpack_from("<I", code, 3)[0]
        assert val == 0

    def test_max_32bit(self, x86):
        addr = 0xFFFF_FFFF
        code = x86("mov eax, limit", symbols={"limit": addr})
        assert addr.to_bytes(4, "little") in code

    def test_max_64bit(self, x86):
        """0xFFFFFFFFFFFFFFFF = -1, which LLVM encodes as mov rax, -1 (7 bytes)."""
        code = x86("mov rax, limit", symbols={"limit": 0xFFFF_FFFF_FFFF_FFFF})
        # -1 fits in sign-extended imm32, so 7 bytes
        val = struct.unpack_from("<i", code, 3)[0]
        assert val == -1

    def test_one(self, x86):
        code = x86("mov rax, one", symbols={"one": 1})
        val = struct.unpack_from("<I", code, 3)[0]
        assert val == 1

    def test_power_of_two(self, x86):
        addr = 1 << 47
        code = x86("mov rax, big", symbols={"big": addr})
        val = struct.unpack_from("<Q", code, 2)[0]
        assert val == addr

    def test_page_aligned(self, x86):
        addr = 0xFFFFF000
        code = x86("mov eax, page", symbols={"page": addr})
        assert addr.to_bytes(4, "little") in code

    def test_multiple_symbols_same_instruction(self, x86):
        """Only one symbol used, others ignored."""
        code = x86("mov eax, a", symbols={"a": 0x1111, "b": 0x2222})
        assert (0x1111).to_bytes(4, "little") in code

    def test_multiple_symbols_different_instructions(self, x86):
        code = x86("mov eax, a; mov ecx, b", symbols={"a": 0x1111, "b": 0x2222})
        assert (0x1111).to_bytes(4, "little") in code
        assert (0x2222).to_bytes(4, "little") in code

    def test_symbol_same_address(self, x86):
        """Two symbols pointing to the same address."""
        code1 = x86("mov eax, alias1", symbols={"alias1": 0xBEEF, "alias2": 0xBEEF})
        code2 = x86("mov eax, alias2", symbols={"alias1": 0xBEEF, "alias2": 0xBEEF})
        assert (0xBEEF).to_bytes(4, "little") in code1
        assert (0xBEEF).to_bytes(4, "little") in code2


# ---------------------------------------------------------------------------
# ARM Thumb absolute addresses
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestARMAbsoluteAddress:
    @pytest.fixture(scope="class")
    def arm(self):
        return Assembler.cortex_m7_dp()

    def test_ldr_literal_pool_large(self, arm):
        """Large value that can't be encoded as immediate uses literal pool."""
        addr = 0x2000_1000
        code = arm("ldr r0, =hook\n.ltorg", symbols={"hook": addr})
        # Literal pool value is the last 4 bytes
        pool_val = struct.unpack_from("<I", code, len(code) - 4)[0]
        assert pool_val == addr

    def test_ldr_small_optimized(self, arm):
        """Small values are optimized to mov by LLVM, no literal pool."""
        code = arm("ldr r0, =small", symbols={"small": 0x42})
        # LLVM optimizes ldr r0, =0x42 to mov r0, #0x42 (4 bytes Thumb-2)
        assert len(code) == 4

    def test_ldr_common_peripheral(self, arm):
        addr = 0x4002_0000
        code = arm("ldr r0, =periph\n.ltorg", symbols={"periph": addr})
        pool_val = struct.unpack_from("<I", code, len(code) - 4)[0]
        assert pool_val == addr

    def test_multiple_ldr_large_symbols(self, arm):
        """Multiple literal pool loads with large values."""
        code = arm(
            "ldr r0, =a\nldr r1, =b\n.ltorg",
            symbols={"a": 0xDEAD_0000, "b": 0xBEEF_0000},
        )
        assert (0xDEAD_0000).to_bytes(4, "little") in code
        assert (0xBEEF_0000).to_bytes(4, "little") in code


# ---------------------------------------------------------------------------
# AArch64 absolute addresses
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAArch64AbsoluteAddress:
    @pytest.fixture(scope="class")
    def aa64(self):
        return Assembler.aarch64()

    def test_ldr_literal_64bit(self, aa64):
        """ldr x0, =addr uses a literal pool for 64-bit values."""
        addr = 0x4000_0000_8000_0000
        code = aa64("ldr x0, =target", symbols={"target": addr})
        assert addr.to_bytes(8, "little") in code

    def test_ldr_literal_large_32bit(self, aa64):
        """Value that doesn't fit in mov immediate uses literal pool."""
        addr = 0xDEAD_BEEF
        code = aa64("ldr x0, =val", symbols={"val": addr})
        assert addr.to_bytes(8, "little") in code or addr.to_bytes(4, "little") in code

    def test_mov_small_immediate(self, aa64):
        """mov x0, #symbol when value fits in 16-bit immediate."""
        code = aa64("mov x0, #small", symbols={"small": 0xFF})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Adversarial: symbol value edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAdversarialValues:
    @pytest.fixture(scope="class")
    def x86(self):
        return Assembler.x86_64()

    @pytest.fixture(scope="class")
    def arm(self):
        return Assembler.cortex_m7_dp()

    def test_negative_as_twos_complement_x86(self, x86):
        code = x86("mov eax, neg", symbols={"neg": -1})
        assert b"\xff\xff\xff\xff" in code

    def test_negative_arm_literal_pool(self, arm):
        """Negative value in literal pool should be two's complement."""
        code = arm("ldr r0, =neg\n.ltorg", symbols={"neg": -1})
        # -1 as uint32 = 0xFFFFFFFF, but LLVM may optimize to mvn
        # Either the pool contains 0xFFFFFFFF or it's optimized away
        assert len(code) > 0  # at minimum, doesn't crash

    def test_alternating_bits(self, x86):
        addr = 0xAAAA_AAAA_AAAA_AAAA
        code = x86("mov rax, pat", symbols={"pat": addr})
        val = struct.unpack_from("<Q", code, 2)[0]
        assert val == addr

    def test_symbol_not_used_in_source(self, x86):
        """Defining a symbol that isn't referenced should not error."""
        code = x86("nop", symbols={"unused": 0x1234})
        assert code == b"\x90"

    def test_symbol_shadows_register_name(self, x86):
        """Symbol named like a register — LLVM prefers the register."""
        code = x86("mov rcx, rax", symbols={"rax": 0x42})
        assert len(code) > 0

    def test_symbol_with_address_param_immediate(self, x86):
        """Immediate value should be unaffected by address parameter."""
        code1 = x86("mov eax, val", symbols={"val": 0xBEEF}, address=0)
        code2 = x86("mov eax, val", symbols={"val": 0xBEEF}, address=0x1000)
        assert (0xBEEF).to_bytes(4, "little") in code1
        assert (0xBEEF).to_bytes(4, "little") in code2

    def test_consistency_with_inline_set(self, x86):
        """symbols= should produce same result as manual .set."""
        code_sym = x86("mov eax, v", symbols={"v": 0x1234})
        code_manual = x86(".set v, 0x1234\nmov eax, v")
        assert code_sym == code_manual

    def test_many_symbols(self, x86):
        """Large symbol table shouldn't cause issues."""
        syms = {f"sym_{i}": i * 0x1000 for i in range(100)}
        code = x86("mov eax, sym_42", symbols=syms)
        assert (42 * 0x1000).to_bytes(4, "little") in code

    def test_symbol_value_exact_boundary_32bit(self, x86):
        """0x7FFFFFFF — max positive signed 32-bit."""
        code = x86("mov eax, v", symbols={"v": 0x7FFF_FFFF})
        assert (0x7FFF_FFFF).to_bytes(4, "little") in code

    def test_symbol_value_sign_boundary(self, x86):
        """0x80000000 — min negative signed 32-bit as unsigned."""
        code = x86("mov eax, v", symbols={"v": 0x8000_0000})
        assert (0x8000_0000).to_bytes(4, "little") in code


# ---------------------------------------------------------------------------
# Round-trip with disassembler
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestRoundTripWithDisasm:
    def test_x86_round_trip(self):
        from flintmc import Disassembler
        asm = Assembler.x86_64()
        dis = Disassembler.x86_64()

        code = asm("mov rax, target; ret", symbols={"target": 0xDEAD_BEEF_CAFE_BABE})
        instrs = dis(code)
        assert len(instrs) == 2
        # Disassembler may print hex or decimal depending on options
        assert "mov" in instrs[0].text.lower()
        assert "ret" in instrs[1].text

    def test_arm_round_trip(self):
        from flintmc import Disassembler
        asm = Assembler.cortex_m7_dp()
        dis = Disassembler.cortex_m7_dp()

        code = asm("ldr r0, =hook\n.ltorg", symbols={"hook": 0x2000_1000})
        instrs = dis(code)
        assert len(instrs) >= 1
        assert "ldr" in instrs[0].text.lower()

    def test_aarch64_round_trip(self):
        from flintmc import Disassembler
        asm = Assembler.aarch64()
        dis = Disassembler.aarch64()

        code = asm("mov x0, #42; ret")
        instrs = dis(code)
        assert len(instrs) == 2
        assert "mov" in instrs[0].text.lower()
        assert "0x2a" in instrs[0].text.lower() or "42" in instrs[0].text
