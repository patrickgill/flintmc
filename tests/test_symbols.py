"""Adversarial and edge-case tests for symbol injection.

Tests injection attacks, invalid names, type errors, boundary values,
interaction with address/cache/asm_each, and cross-architecture behavior.
"""

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
