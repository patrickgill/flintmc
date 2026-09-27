"""Tests for features added during the code audit.

Covers: Mach-O extraction, COFF extraction, big-endian ELF, _split_semicolons
edge cases, _default_preamble, _fix_error, _extract_text dispatch,
register_default_preamble, register_arch_mapping, close()/context manager,
__repr__, asm_each(), address parameter, InstructionInfo.
"""

import struct
import pytest
import flintmc
from flintmc import Assembler, AsmError, InstructionInfo
from flintmc.common import (
    _split_semicolons,
    _default_preamble,
    _fix_error,
    register_default_preamble,
    _PREAMBLE_OVERRIDES,
)
from flintmc.objfile import extract_text as _extract_text, _elf_endian
import flintmc.llvm_capi
from flintmc.llvm_capi import register_arch_mapping


# ---------------------------------------------------------------------------
# _split_semicolons
# ---------------------------------------------------------------------------

class TestSplitSemicolons:
    def test_no_semicolons(self):
        assert _split_semicolons("nop") == "nop"

    def test_basic_split(self):
        assert _split_semicolons("nop; ret") == "nop\n ret"

    def test_double_quotes(self):
        assert _split_semicolons('mov al, ";"') == 'mov al, ";"'

    def test_single_quotes(self):
        assert _split_semicolons("mov al, ';'") == "mov al, ';'"

    def test_apostrophe_inside_double_quotes(self):
        assert _split_semicolons('.ascii "it\'s; ok"') == '.ascii "it\'s; ok"'

    def test_escaped_quote_inside_string(self):
        assert _split_semicolons('.ascii "a\\"b; c"') == '.ascii "a\\"b; c"'

    def test_hash_immediate_digit(self):
        result = _split_semicolons("mov x0, #42; ret")
        assert "ret" in result
        assert "\n" in result

    def test_hash_immediate_negative(self):
        result = _split_semicolons("mov x0, #-1; ret")
        assert "\n" in result

    def test_hash_immediate_paren(self):
        result = _split_semicolons("mov x0, #(1 << 5); ret")
        assert "\n" in result

    def test_hash_comment(self):
        result = _split_semicolons("nop # comment; not split")
        assert "\n" not in result

    def test_hash_end_of_line(self):
        result = _split_semicolons("nop #")
        assert result == "nop #"

    def test_at_comment(self):
        result = _split_semicolons("add r0, r1 @ comment; not split")
        assert "\n" not in result

    def test_c_style_comment(self):
        result = _split_semicolons("nop // comment; not split")
        assert "\n" not in result

    def test_block_comment(self):
        result = _split_semicolons("nop /* ; */ ret")
        assert ";" not in result or result.count("\n") == 0
        # The semicolon is inside a block comment, so no split

    def test_multiline_preserves_newlines(self):
        result = _split_semicolons("nop\nret")
        assert result == "nop\nret"

    def test_multiple_semicolons(self):
        result = _split_semicolons("a; b; c")
        parts = result.split("\n")
        assert len(parts) == 3


# ---------------------------------------------------------------------------
# _default_preamble
# ---------------------------------------------------------------------------

class TestDefaultPreamble:
    def test_thumb(self):
        assert _default_preamble("thumbv7em-none-eabi") == ".syntax unified\n.thumb"

    def test_armv7m(self):
        assert _default_preamble("armv7m-none-eabi") == ".syntax unified\n.thumb"

    def test_armv8m_main(self):
        assert _default_preamble("armv8m.main-none-eabi") == ".syntax unified\n.thumb"

    def test_armv7a_gets_arm_not_thumb(self):
        assert _default_preamble("armv7a-none-eabi") == ".syntax unified\n.arm"

    def test_armv8a_gets_arm_not_thumb(self):
        assert _default_preamble("armv8a-none-eabi") == ".syntax unified\n.arm"

    def test_arm_generic(self):
        assert _default_preamble("arm-none-eabi") == ".syntax unified\n.arm"

    def test_arm64_no_preamble(self):
        assert _default_preamble("arm64-apple-macos") == ""

    def test_aarch64_no_preamble(self):
        assert _default_preamble("aarch64") == ""

    def test_x86_64(self):
        assert _default_preamble("x86_64") == ".intel_syntax noprefix"

    def test_x86_dash_64(self):
        assert _default_preamble("x86-64") == ".intel_syntax noprefix"

    def test_i686(self):
        p = _default_preamble("i686")
        assert ".intel_syntax noprefix" in p
        assert ".code32" in p

    def test_riscv_no_preamble(self):
        assert _default_preamble("riscv64") == ""

    def test_unknown_no_preamble(self):
        assert _default_preamble("unknown-triple") == ""


# ---------------------------------------------------------------------------
# register_default_preamble
# ---------------------------------------------------------------------------

class TestRegisterDefaultPreamble:
    def setup_method(self):
        self._orig = dict(_PREAMBLE_OVERRIDES)

    def teardown_method(self):
        _PREAMBLE_OVERRIDES.clear()
        _PREAMBLE_OVERRIDES.update(self._orig)

    def test_basic(self):
        register_default_preamble("mycpu", ".option foo")
        assert _default_preamble("mycpu-none-elf") == ".option foo"

    def test_override_builtin(self):
        register_default_preamble("x86_64", ".att_syntax")
        assert _default_preamble("x86_64") == ".att_syntax"

    def test_longest_prefix_wins(self):
        register_default_preamble("myc", "short")
        register_default_preamble("mycpu", "long")
        assert _default_preamble("mycpu-none") == "long"


# ---------------------------------------------------------------------------
# register_arch_mapping
# ---------------------------------------------------------------------------

class TestRegisterArchMapping:
    def setup_method(self):
        self._added: list[str] = []

    def teardown_method(self):
        mapping = flintmc.llvm_capi._TRIPLE_TO_ARCH
        for key in self._added:
            mapping.pop(key, None)

    def test_basic(self):
        register_arch_mapping("fakearch", "FakeArch")
        self._added.append("fakearch")
        assert flintmc.llvm_capi._TRIPLE_TO_ARCH["fakearch"] == "FakeArch"

    def test_case_insensitive(self):
        register_arch_mapping("FooBar", "Foo")
        self._added.append("foobar")
        assert "foobar" in flintmc.llvm_capi._TRIPLE_TO_ARCH


# ---------------------------------------------------------------------------
# _fix_error
# ---------------------------------------------------------------------------

class TestFixError:
    def test_stdin_line_adjustment(self):
        err = "<stdin>:5: error: bad instruction"
        result = _fix_error(err, preamble_lines=2)
        assert "line 3:" in result

    def test_inline_asm_line_adjustment(self):
        err = "<inline asm>:3: error: bad"
        result = _fix_error(err, preamble_lines=1)
        assert "line 2:" in result

    def test_no_match_passthrough(self):
        err = "some other error"
        assert _fix_error(err, preamble_lines=2) == err

    def test_min_line_1(self):
        err = "<stdin>:1: error: bad"
        result = _fix_error(err, preamble_lines=5)
        assert "line 1:" in result

    def test_multiline(self):
        err = "<stdin>:3: first\n<stdin>:4: second"
        result = _fix_error(err, preamble_lines=1)
        assert "line 2:" in result
        assert "line 3:" in result


# ---------------------------------------------------------------------------
# _extract_text dispatch and error paths
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_too_short(self):
        with pytest.raises(AsmError, match="valid object output"):
            _extract_text(b"\x00")

    def test_bad_magic(self):
        with pytest.raises(AsmError, match="valid ELF, Mach-O, or COFF"):
            _extract_text(b"\x00" * 20)

    def test_truncated_elf(self):
        with pytest.raises(AsmError, match="truncated"):
            _extract_text(b"\x7fELF\x01")

    def test_bad_elf_class(self):
        with pytest.raises(AsmError, match="unsupported ELF class"):
            _extract_text(b"\x7fELF\x03\x01")

    def test_bad_elf_endian(self):
        with pytest.raises(AsmError, match="unsupported ELF endianness"):
            _elf_endian(b"\x7fELF\x01\x03")


# ---------------------------------------------------------------------------
# Mach-O extraction (requires C API on macOS)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestMachO:
    @pytest.mark.parametrize("triple, source, expected", [
        # Mach-O left these as unapplied relocations (e9 00000000 etc.)
        ("x86_64-apple-macos", "jmp end; nop; end: ret", "eb0190c3"),
        ("arm64-apple-macos", "b end; nop; end: ret", "020000141f2003d5c0035fd6"),
        ("arm64-apple-macos", "cbz x0, e; e: ret", "200000b4c0035fd6"),
        ("arm64-apple-macos", "adr x0, e; e: ret", "20000010c0035fd6"),
    ])
    def test_local_label_branches(self, triple, source, expected):
        assert Assembler(triple).asm(source).hex() == expected

    def test_arm64_apple_triple(self):
        asm = Assembler(triple="arm64-apple-macos")
        code = asm("ret")
        assert len(code) == 4
        assert code == b"\xc0\x03\x5f\xd6"

    def test_arm64_apple_multi_instruction(self):
        asm = Assembler(triple="arm64-apple-macos")
        code = asm("mov x0, #42\nret")
        assert len(code) == 8

    def test_arm64_apple_empty(self):
        asm = Assembler(triple="arm64-apple-macos")
        assert asm("") == b""


# ---------------------------------------------------------------------------
# COFF extraction (requires C API)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestCOFF:
    def test_windows_triple(self):
        asm = Assembler(triple="x86_64-pc-windows-msvc")
        code = asm("nop; ret")
        assert code == b"\x90\xc3"

    def test_windows_empty(self):
        asm = Assembler(triple="x86_64-pc-windows-msvc")
        assert asm("") == b""

    def test_windows_symbol_branch(self):
        # LLVM crashed emitting jmp-to-.set with COFF output; now assembled as ELF
        asm = Assembler("x86_64-pc-windows-msvc", preamble="")
        assert asm("jmp target", symbols={"target": 0x8000}) == bytes.fromhex("e9fb7f0000")


# ---------------------------------------------------------------------------
# close() and context manager
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestCloseAndContextManager:
    def test_close_and_reuse(self):
        asm = Assembler.x86_64()
        code1 = asm("nop")
        asm.close()
        code2 = asm("nop")
        assert code1 == code2

    def test_double_close(self):
        asm = Assembler.x86_64()
        asm("nop")
        asm.close()
        asm.close()  # should not raise

    def test_context_manager(self):
        with Assembler.x86_64() as asm:
            code = asm("nop; ret")
        assert code == b"\x90\xc3"

    def test_cache_cleared_on_close(self):
        asm = Assembler.x86_64()
        asm("nop")
        assert asm.cache_size > 0
        asm.close()
        assert asm.cache_size == 0


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_basic(self):
        asm = Assembler.x86_64()
        r = repr(asm)
        assert "x86_64" in r
        assert "backend=" in r

    def test_with_cpu(self):
        asm = Assembler.cortex_m7_dp()
        r = repr(asm)
        assert "cortex-m7" in r


# ---------------------------------------------------------------------------
# asm_each()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAsmEach:
    def test_basic(self):
        asm = Assembler.x86_64()
        results = asm.asm_each("mov eax, 1; nop; ret")
        assert len(results) == 3
        assert results[0].source == "mov eax, 1"
        assert results[0].offset == 0
        assert results[0].size == 5
        assert results[1].source == "nop"
        assert results[1].offset == 5
        assert results[1].size == 1
        assert results[2].source == "ret"
        assert results[2].offset == 6
        assert results[2].size == 1

    def test_with_labels(self):
        asm = Assembler.aarch64()
        results = asm.asm_each("loop:\n  mov x0, #42\n  b loop")
        assert len(results) == 2
        assert results[0].source == "mov x0, #42"
        assert results[1].source == "b loop"

    def test_concatenated_bytes_match_asm(self):
        asm = Assembler.x86_64()
        source = "push rbp; mov rbp, rsp; pop rbp; ret"
        each = asm.asm_each(source)
        full = asm(source)
        concat = b"".join(info.code for info in each)
        assert concat == full

    def test_empty(self):
        asm = Assembler.x86_64()
        assert asm.asm_each("") == []

    @pytest.mark.parametrize("profile, source", [
        (Assembler.x86_64, "jmp end; nop; end: ret"),         # relaxes to short jmp
        (Assembler.cortex_m4, "b end; nop; end: bx lr"),
        (Assembler.riscv64, "j end; nop; end: ret"),
    ])
    def test_forward_reference_matches_asm(self, profile, source):
        asm = profile()
        each = asm.asm_each(source)
        assert len(each) == 3
        assert b"".join(i.code for i in each) == asm(source)

    def test_literal_pool_not_attributed_to_ldr(self):
        each = Assembler.cortex_m4().asm_each("ldr r0, =0x12345678; bx lr")
        assert [(i.offset, i.size, i.source) for i in each] == [
            (0, 2, "ldr r0, =0x12345678"), (2, 2, "bx lr")]

    @pytest.mark.parametrize("kw", [{"address": 0x100}, {"symbols": {"t": 0x8000}},
                                    {"symbols": {"t": 0x8000}, "address": 0x1000}])
    def test_address_and_symbols_match_asm(self, kw):
        asm = Assembler("x86_64", preamble="")  # AT&T: symbol branches
        source = "nop; jmp t; call t" if "symbols" in kw else "nop; lea 1f(%rip), %rax; 1: ret"
        each = asm.asm_each(source, **kw)
        assert b"".join(i.code for i in each) == asm(source, **kw)
        assert each[0].offset == 0

    def test_address_validated(self):
        with pytest.raises(ValueError):
            Assembler.x86_64().asm_each("nop", address=-1)

    def test_error_line_and_column(self):
        with pytest.raises(AsmError, match=r"^line 3:1: "):
            Assembler.x86_64().asm_each("nop\nnop\nbogus")

    def test_instruction_info_frozen(self):
        asm = Assembler.x86_64()
        results = asm.asm_each("nop")
        with pytest.raises(AttributeError):
            results[0].offset = 99


# ---------------------------------------------------------------------------
# address parameter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestAddress:
    def test_basic(self):
        asm = Assembler.aarch64()
        # adr x0, . is PC-relative to self — same regardless of address
        code0 = asm("adr x0, .", address=0)
        code100 = asm("adr x0, .", address=0x100)
        assert code0 == code100

    def test_address_zero_same_as_default(self):
        asm = Assembler.x86_64()
        default = asm("nop; ret")
        at_zero = asm("nop; ret", address=0)
        assert default == at_zero

    def test_negative_address_raises(self):
        asm = Assembler.x86_64()
        with pytest.raises(ValueError, match="non-negative"):
            asm("nop", address=-1)

    def test_too_large_address_raises(self):
        asm = Assembler.x86_64()
        with pytest.raises(ValueError, match="below 2\\*\\*64"):
            asm("nop", address=2**64)

    def test_large_address_aarch64_adrp(self):
        from flintmc import Disassembler
        # t sits at 0x0800_0800 + 0x1ff8 = 0x0800_27f8 -> page delta 0x2000, lo12 0x7f8
        code = Assembler.aarch64().asm("adrp x0, t; ldr x1, [x0, :lo12:t]; .space 0x1ff0; t: .quad 1",
                                       address=0x0800_0800)
        adrp, ldr = (i.text.replace("\t", " ") for i in Disassembler.aarch64()(code[:8]))
        assert (adrp, ldr) == ("adrp x0, #8192", "ldr x1, [x0, #0x7f8]")

    def test_large_address_values(self):
        asm = Assembler("x86_64", preamble="")
        code = asm("call h", symbols={"h": 0x0800_0100}, address=0x0800_0000)
        assert code == b"\xe8" + (0x100 - 5).to_bytes(4, "little")
        code = asm("movabsq $l, %rax; l: ret", address=0xFFFF_0000_1234)
        assert code[2:10] == (0xFFFF_0000_1234 + 10).to_bytes(8, "little")
        code = asm("nop; .p2align 4; ret", address=0x0800_1001)
        assert len(code) == 16  # 1 nop + pad to 0x08001010 + ret

    def test_large_address_is_fast(self):
        import time
        asm = Assembler.cortex_m4()
        t = time.perf_counter()
        asm("bl h; ldr r0, =h", symbols={"h": 0x0800_0100}, address=0x0800_0000)
        assert time.perf_counter() - t < 0.2  # was ~0.6 s and 400 MB via full .org

    def test_via_call(self):
        asm = Assembler.x86_64()
        code = asm("nop", address=0x100)
        assert code == b"\x90"

    def test_not_cached(self):
        """address= calls bypass cache so different addresses don't collide."""
        asm = Assembler.x86_64()
        asm("nop", address=0x100)
        # Cache should not have grown (address calls bypass cache)
        prev = asm.cache_size
        asm("nop", address=0x200)
        assert asm.cache_size == prev


# ---------------------------------------------------------------------------
# arm64 prefix resolution
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestArm64Prefix:
    def test_arm64_triple(self):
        asm = Assembler(triple="arm64")
        code = asm("ret")
        assert len(code) == 4

    def test_arm64_matches_aarch64(self):
        asm1 = Assembler(triple="arm64")
        asm2 = Assembler.aarch64()
        assert asm1("ret") == asm2("ret")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

class TestModuleLevel:
    def test_no_default_raises(self):
        saved = flintmc.default
        try:
            flintmc.default = None
            with pytest.raises(RuntimeError, match="No default"):
                flintmc.asm("nop")
        finally:
            flintmc.default = saved

    def test_with_default(self):
        saved = flintmc.default
        try:
            flintmc.default = Assembler.x86_64()
            assert flintmc.asm("nop") == b"\x90"
        finally:
            flintmc.default = saved


# ---------------------------------------------------------------------------
# verify=True (round-trip validation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API not available")
class TestVerify:
    def test_x86_passes(self):
        asm = Assembler.x86_64()
        code = asm("nop; ret", verify=True)
        assert code == b"\x90\xc3"

    def test_aarch64_passes(self):
        asm = Assembler.aarch64()
        code = asm("mov x0, #42; ret", verify=True)
        assert len(code) == 8

    def test_arm_thumb_passes(self):
        asm = Assembler.cortex_m7_dp()
        code = asm("nop; bx lr", verify=True)
        assert len(code) == 4

    def test_empty_passes(self):
        asm = Assembler.x86_64()
        assert asm("", verify=True) == b""

    def test_complex_x86(self):
        asm = Assembler.x86_64()
        code = asm("push rbp; mov rbp, rsp; pop rbp; ret", verify=True)
        assert len(code) > 0

    def test_via_call(self):
        asm = Assembler.x86_64()
        code = asm("nop", verify=True)
        assert code == b"\x90"

    def test_cached_result_also_verified(self):
        asm = Assembler.x86_64()
        asm("nop")  # prime the cache
        # second call hits cache but verify still runs
        code = asm("nop", verify=True)
        assert code == b"\x90"

    def test_with_symbols(self):
        asm = Assembler.aarch64()
        code = asm("mov x0, #val", symbols={"val": 0xFF}, verify=True)
        assert len(code) == 4

    def test_with_address(self):
        asm = Assembler.x86_64()
        code = asm("nop; ret", address=0x100, verify=True)
        assert code == b"\x90\xc3"

    def test_multi_instruction_x86(self):
        asm = Assembler.x86_64()
        code = asm("mov eax, 1; xor ecx, ecx; syscall", verify=True)
        assert len(code) > 0

    def test_riscv_passes(self):
        asm = Assembler.riscv64()
        code = asm("addi x1, x0, 42", verify=True)
        assert len(code) == 4

    @pytest.mark.parametrize("source", [
        "top: nop; jmp top; call top; jne top; loop top",
        "jmp e; nop; e: ret",
    ])
    def test_x86_relative_branches(self, source):
        asm = Assembler.x86_64()
        assert asm(source, verify=True) == asm(source)

    def test_att_syntax(self):
        asm = Assembler("x86_64", preamble="")
        assert asm("top: movl $1, %eax; jmp top", verify=True) == bytes.fromhex("b801000000ebf9")

    def test_default_verify(self, monkeypatch):
        asm = Assembler.x86_64()
        calls = []
        monkeypatch.setattr(asm, "_verify_round_trip", calls.append)
        asm("nop")
        assert calls == []
        monkeypatch.setattr(Assembler, "default_verify", True)
        asm("ret")
        asm("nop", verify=False)
        assert calls == [b"\xc3"]
