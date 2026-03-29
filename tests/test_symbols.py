"""Adversarial and edge-case tests for symbol injection.

Tests injection attacks, invalid names, type errors, boundary values,
interaction with address/cache/asm_each, cross-architecture behavior,
and absolute address encoding verification.
"""

import struct
import pytest
from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def aarch64():
    return Assembler.aarch64()


@pytest.fixture(scope="module")
def x86():
    return Assembler.x86_64()


@pytest.fixture(scope="module")
def arm():
    return Assembler.cortex_m7_dp()


# ---------------------------------------------------------------------------
# Injection attacks
# ---------------------------------------------------------------------------

class TestInjectionAttacks:
    """Attempt to inject assembly via crafted symbol names."""

    def test_newline_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo\nnop": 0x1000})

    def test_semicolon_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo;nop": 0x1000})

    def test_directive_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={".set evil, 0": 0x1000})

    def test_space_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo bar": 0x1000})

    def test_tab_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo\tbar": 0x1000})

    def test_null_byte_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo\x00bar": 0x1000})

    def test_comma_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"foo,0x1234": 0x1000})

    def test_hash_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"#immediate": 0x1000})

    def test_at_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"sym@PLT": 0x1000})

    def test_slash_in_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"path/to/sym": 0x1000})

    def test_empty_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"": 0x1000})

    def test_starts_with_digit(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"0bad": 0x1000})

    def test_unicode_name(self, aarch64):
        with pytest.raises(ValueError, match="Invalid symbol name"):
            aarch64("nop", symbols={"caf\u00e9": 0x1000})


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

class TestTypeErrors:
    def test_string_address(self, aarch64):
        with pytest.raises(TypeError, match="must be int"):
            aarch64("nop", symbols={"foo": "0x1000"})

    def test_float_address(self, aarch64):
        with pytest.raises(TypeError, match="must be int"):
            aarch64("nop", symbols={"foo": 3.14})

    def test_none_address(self, aarch64):
        with pytest.raises(TypeError, match="must be int"):
            aarch64("nop", symbols={"foo": None})


# ---------------------------------------------------------------------------
# Valid symbol names (edge cases that should pass)
# ---------------------------------------------------------------------------

class TestValidNames:
    def test_underscore_prefix(self, aarch64):
        code = aarch64("bl _start", symbols={"_start": 0x100})
        assert len(code) == 4

    def test_dot_prefix(self, aarch64):
        code = aarch64("bl .Llocal", symbols={".Llocal": 0x100})
        assert len(code) == 4

    def test_dollar_in_name(self, aarch64):
        code = aarch64("bl my$func", symbols={"my$func": 0x100})
        assert len(code) == 4

    def test_long_name(self, aarch64):
        name = "a" * 200
        code = aarch64(f"bl {name}", symbols={name: 0x100})
        assert len(code) == 4

    def test_single_char(self, aarch64):
        code = aarch64("bl x", symbols={"x": 0x100})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Address boundary values
# ---------------------------------------------------------------------------

class TestSymbolValues:
    def test_zero_value(self, aarch64):
        code = aarch64("bl target", symbols={"target": 0})
        assert len(code) == 4

    def test_large_immediate(self, x86):
        code = x86("mov rax, big_val", symbols={"big_val": 0xDEAD_BEEF_CAFE_BABE})
        assert len(code) > 0

    def test_negative_value(self, aarch64):
        # Negative ints are valid in .set (two's complement)
        code = aarch64("bl target", symbols={"target": -4})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Cross-architecture
# ---------------------------------------------------------------------------

class TestCrossArch:
    def test_aarch64_bl(self, aarch64):
        code = aarch64("bl handler", symbols={"handler": 0x1000})
        assert len(code) == 4

    def test_arm_thumb_ldr(self, arm):
        code = arm("ldr r0, =hook", symbols={"hook": 0x20001000})
        assert len(code) > 0

    def test_x86_mov_immediate(self, x86):
        code = x86("mov rax, my_var", symbols={"my_var": 0xDEADBEEF})
        assert 0xDEADBEEF.to_bytes(4, "little") in code

    def test_riscv(self):
        rv = Assembler.riscv64()
        code = rv("jal ra, target", symbols={"target": 0x1000})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Interaction with other features
# ---------------------------------------------------------------------------

class TestInteraction:
    def test_symbols_with_address(self, aarch64):
        """Symbols + address should both work together."""
        # .set values are section-relative offsets for branches, so the
        # encoding is the same regardless of address (address only affects
        # .org padding, not symbol resolution)
        code = aarch64("bl handler", symbols={"handler": 0x100}, address=0x80)
        assert len(code) == 4

    def test_symbols_bypass_cache(self, aarch64):
        """Symbol calls should not pollute the cache."""
        size_before = aarch64.cache_size
        aarch64("bl sym", symbols={"sym": 0x100})
        assert aarch64.cache_size == size_before

    def test_multiple_symbols(self, aarch64):
        code = aarch64("bl foo\nbl bar", symbols={"foo": 0x100, "bar": 0x200})
        assert len(code) == 8

    def test_symbol_overrides_label(self, aarch64):
        """A .set symbol should override a label of the same name."""
        # The .set directive comes before the code, so 'target' resolves
        # to the external address, not the label
        code = aarch64("bl target", symbols={"target": 0x1000})
        code_with_label = aarch64("target:\nbl target")
        assert code != code_with_label

    def test_empty_symbols_dict(self, aarch64):
        """Empty dict should behave like no symbols."""
        code1 = aarch64("nop")
        code2 = aarch64("nop", symbols={})
        assert code1 == code2

    def test_via_call_syntax(self, aarch64):
        """symbols works via __call__ too."""
        code = aarch64("bl handler", symbols={"handler": 0x1000})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# x86 branch relocation (R_X86_64_PC32)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestX86BranchRelocation:
    """Verify that x86 branch-to-symbol resolves via relocation patching."""

    @pytest.fixture(scope="class")
    def x86_att(self):
        return Assembler(triple="x86_64", preamble="")

    def test_jmp_absolute(self, x86_att):
        code = x86_att("jmp target", symbols={"target": 0x8000})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 5 + offset == 0x8000

    def test_call_absolute(self, x86_att):
        code = x86_att("call target", symbols={"target": 0x4000})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 5 + offset == 0x4000

    def test_jmp_with_address(self, x86_att):
        """Branch from address=0x1000 to target=0x8000."""
        code = x86_att("jmp target", symbols={"target": 0x8000}, address=0x1000)
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 0x1000 + 5 + offset == 0x8000

    def test_call_with_address(self, x86_att):
        code = x86_att("call target", symbols={"target": 0x8000}, address=0x1000)
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 0x1000 + 5 + offset == 0x8000

    def test_multiple_branches(self, x86_att):
        """Two branches to the same target get correct independent offsets."""
        code = x86_att("jmp target\ncall target", symbols={"target": 0x8000})
        off1 = struct.unpack_from("<i", code, 1)[0]
        off2 = struct.unpack_from("<i", code, 6)[0]
        assert 5 + off1 == 0x8000
        assert 10 + off2 == 0x8000

    def test_multiple_targets(self, x86_att):
        code = x86_att("jmp a\njmp b", symbols={"a": 0x1000, "b": 0x2000})
        off1 = struct.unpack_from("<i", code, 1)[0]
        off2 = struct.unpack_from("<i", code, 6)[0]
        assert 5 + off1 == 0x1000
        assert 10 + off2 == 0x2000

    def test_backward_branch(self, x86_att):
        """Branch to address 0 (before the code)."""
        code = x86_att("jmp target", symbols={"target": 0}, address=0x100)
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 0x100 + 5 + offset == 0

    def test_no_relocation_without_symbols(self, x86_att):
        """Without symbols param, .text bytes are returned as-is (no patching)."""
        code = x86_att("nop")
        assert code == b"\x90"


# ---------------------------------------------------------------------------
# Adversarial relocation tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAdversarialRelocations:
    """Edge cases and stress tests for ELF relocation patching."""

    @pytest.fixture(scope="class")
    def x86_att(self):
        return Assembler(triple="x86_64", preamble="")

    def test_target_zero(self, x86_att):
        """Branch to address 0."""
        code = x86_att("jmp target", symbols={"target": 0})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 5 + offset == 0

    def test_target_negative(self, x86_att):
        """Negative target (two's complement)."""
        code = x86_att("jmp target", symbols={"target": -16})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 5 + offset == -16

    def test_target_max_positive_32bit(self, x86_att):
        """Branch to near the 32-bit signed limit."""
        code = x86_att("call target", symbols={"target": 0x7FFF_FFF0})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert 5 + offset == 0x7FFF_FFF0

    def test_branch_to_self(self, x86_att):
        """jmp to address 0 is a backward branch to before the instruction."""
        code = x86_att("jmp self", symbols={"self": 0})
        offset = struct.unpack_from("<i", code, 1)[0]
        assert offset == -5  # PC is at 5 after jmp, -5 gets back to 0

    def test_conditional_branch(self, x86_att):
        """Conditional branches (je/jne) also use PC32 relocations."""
        code = x86_att("testl %eax, %eax\nje target", symbols={"target": 0x200})
        # testl is 2 bytes, je near is 6 bytes (0f 84 xx xx xx xx)
        assert len(code) == 8
        offset = struct.unpack_from("<i", code, 4)[0]
        assert 8 + offset == 0x200

    def test_mixed_branch_and_data(self, x86_att):
        """Branch and data reference to different symbols in same block."""
        code = x86_att(
            "jmp handler\nmovq $0, addr",
            symbols={"handler": 0x1000, "addr": 0xBEEF},
        )
        # jmp should resolve correctly
        off = struct.unpack_from("<i", code, 1)[0]
        assert 5 + off == 0x1000
        # addr should appear as an absolute value in the mov
        assert (0xBEEF).to_bytes(4, "little") in code

    def test_unused_symbol_no_effect(self, x86_att):
        """Extra symbols that aren't referenced don't affect output."""
        code1 = x86_att("jmp a", symbols={"a": 0x100})
        code2 = x86_att("jmp a", symbols={"a": 0x100, "b": 0xDEAD, "c": 0xBEEF})
        assert code1 == code2

    def test_symbol_overrides_label(self, x86_att):
        """.set symbol takes precedence over a local label."""
        code_sym = x86_att("jmp target", symbols={"target": 0x500})
        code_label = x86_att("target: nop\njmp target")
        assert code_sym != code_label

    def test_large_code_block(self, x86_att):
        """Many instructions with a branch at the end."""
        nops = "\n".join(["nop"] * 100)
        code = x86_att(f"{nops}\njmp target", symbols={"target": 0x8000})
        # 100 nops (1 byte each) + jmp (5 bytes)
        assert len(code) == 105
        offset = struct.unpack_from("<i", code, 101)[0]
        assert 105 + offset == 0x8000

    def test_branch_with_address_and_symbols(self, x86_att):
        """Triple combo: address + symbols + branch relocation."""
        code = x86_att(
            "jmp handler",
            symbols={"handler": 0x8000},
            address=0x1000,
        )
        offset = struct.unpack_from("<i", code, 1)[0]
        # Code is at 0x1000, jmp imm32 starts at 0x1001, RIP after = 0x1005
        assert 0x1000 + 5 + offset == 0x8000

    def test_two_calls_same_target(self, x86_att):
        """Two calls to same target get different offsets (different PC)."""
        code = x86_att("call f\ncall f", symbols={"f": 0x1000})
        off1 = struct.unpack_from("<i", code, 1)[0]
        off2 = struct.unpack_from("<i", code, 6)[0]
        assert 5 + off1 == 0x1000
        assert 10 + off2 == 0x1000
        assert off1 != off2  # different PC means different offset

    def test_intel_syntax_data_still_works(self):
        """Intel syntax doesn't support jmp to .set symbols, but data refs work."""
        x86 = Assembler.x86_64()
        code = x86("mov rax, my_func", symbols={"my_func": 0xDEAD_BEEF})
        assert (0xDEAD_BEEF).to_bytes(4, "little") in code


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
        val = struct.unpack_from("<I", code, 3)[0]
        assert val == 0

    def test_max_32bit(self, x86):
        addr = 0xFFFF_FFFF
        code = x86("mov eax, limit", symbols={"limit": addr})
        assert addr.to_bytes(4, "little") in code

    def test_max_64bit(self, x86):
        """0xFFFFFFFFFFFFFFFF = -1, which LLVM encodes as mov rax, -1 (7 bytes)."""
        code = x86("mov rax, limit", symbols={"limit": 0xFFFF_FFFF_FFFF_FFFF})
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
        code = x86("mov eax, a", symbols={"a": 0x1111, "b": 0x2222})
        assert (0x1111).to_bytes(4, "little") in code

    def test_multiple_symbols_different_instructions(self, x86):
        code = x86("mov eax, a; mov ecx, b", symbols={"a": 0x1111, "b": 0x2222})
        assert (0x1111).to_bytes(4, "little") in code
        assert (0x2222).to_bytes(4, "little") in code

    def test_symbol_same_address(self, x86):
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
        pool_val = struct.unpack_from("<I", code, len(code) - 4)[0]
        assert pool_val == addr

    def test_ldr_small_optimized(self, arm):
        """Small values are optimized to mov by LLVM, no literal pool."""
        code = arm("ldr r0, =small", symbols={"small": 0x42})
        assert len(code) == 4

    def test_ldr_common_peripheral(self, arm):
        addr = 0x4002_0000
        code = arm("ldr r0, =periph\n.ltorg", symbols={"periph": addr})
        pool_val = struct.unpack_from("<I", code, len(code) - 4)[0]
        assert pool_val == addr

    def test_multiple_ldr_large_symbols(self, arm):
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
        addr = 0x4000_0000_8000_0000
        code = aa64("ldr x0, =target", symbols={"target": addr})
        assert addr.to_bytes(8, "little") in code

    def test_ldr_literal_large_32bit(self, aa64):
        addr = 0xDEAD_BEEF
        code = aa64("ldr x0, =val", symbols={"val": addr})
        assert addr.to_bytes(8, "little") in code or addr.to_bytes(4, "little") in code

    def test_mov_small_immediate(self, aa64):
        code = aa64("mov x0, #small", symbols={"small": 0xFF})
        assert len(code) == 4


# ---------------------------------------------------------------------------
# Adversarial: absolute address edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAdversarialAbsoluteValues:
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
        code = arm("ldr r0, =neg\n.ltorg", symbols={"neg": -1})
        assert len(code) > 0

    def test_alternating_bits(self, x86):
        addr = 0xAAAA_AAAA_AAAA_AAAA
        code = x86("mov rax, pat", symbols={"pat": addr})
        val = struct.unpack_from("<Q", code, 2)[0]
        assert val == addr

    def test_symbol_not_used_in_source(self, x86):
        code = x86("nop", symbols={"unused": 0x1234})
        assert code == b"\x90"

    def test_symbol_shadows_register_name(self, x86):
        code = x86("mov rcx, rax", symbols={"rax": 0x42})
        assert len(code) > 0

    def test_symbol_with_address_param_immediate(self, x86):
        code1 = x86("mov eax, val", symbols={"val": 0xBEEF}, address=0)
        code2 = x86("mov eax, val", symbols={"val": 0xBEEF}, address=0x1000)
        assert (0xBEEF).to_bytes(4, "little") in code1
        assert (0xBEEF).to_bytes(4, "little") in code2

    def test_consistency_with_inline_set(self, x86):
        code_sym = x86("mov eax, v", symbols={"v": 0x1234})
        code_manual = x86(".set v, 0x1234\nmov eax, v")
        assert code_sym == code_manual

    def test_many_symbols(self, x86):
        syms = {f"sym_{i}": i * 0x1000 for i in range(100)}
        code = x86("mov eax, sym_42", symbols=syms)
        assert (42 * 0x1000).to_bytes(4, "little") in code

    def test_symbol_value_exact_boundary_32bit(self, x86):
        code = x86("mov eax, v", symbols={"v": 0x7FFF_FFFF})
        assert (0x7FFF_FFFF).to_bytes(4, "little") in code

    def test_symbol_value_sign_boundary(self, x86):
        code = x86("mov eax, v", symbols={"v": 0x8000_0000})
        assert (0x8000_0000).to_bytes(4, "little") in code


# ---------------------------------------------------------------------------
# Round-trip: assemble with symbols, then disassemble
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestSymbolRoundTrip:
    def test_x86_round_trip(self):
        from flintmc import Disassembler
        asm = Assembler.x86_64()
        dis = Disassembler.x86_64()

        code = asm("mov rax, target; ret", symbols={"target": 0xDEAD_BEEF_CAFE_BABE})
        instrs = dis(code)
        assert len(instrs) == 2
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
