"""Tests for Thumb-2 (ARM Cortex-M) assembly."""

import pytest

from flintmc import Assembler, AsmError


@pytest.fixture(scope="module")
def asm():
    return Assembler.cortex_m7_dp()


# ---------------------------------------------------------------------------
# Basic Thumb-2
# ---------------------------------------------------------------------------

class TestBasicThumb:
    def test_nop(self, asm):
        assert asm.asm("nop") == b"\x00\xbf"

    def test_mov_immediate(self, asm):
        code = asm.asm("mov r0, #0")
        assert len(code) in (2, 4)

    def test_bx_lr(self, asm):
        assert asm.asm("bx lr") == b"\x70\x47"

    def test_multi_instructions(self, asm):
        code = asm.asm("mov r0, #0; bx lr")
        assert len(code) >= 4  # at least 2 instructions

    def test_multi_newlines(self, asm):
        code = asm.asm("mov r0, #0\nbx lr")
        assert len(code) >= 4


# ---------------------------------------------------------------------------
# Cortex-M special registers (the keystone gaps)
# ---------------------------------------------------------------------------

class TestMClassRegisters:
    def test_mrs_primask(self, asm):
        assert asm.asm("mrs r0, PRIMASK") == bytes.fromhex("eff31080")

    def test_msr_basepri(self, asm):
        assert asm.asm("msr BASEPRI, r1") == bytes.fromhex("81f31188")

    def test_mrs_faultmask(self, asm):
        assert len(asm.asm("mrs r0, FAULTMASK")) == 4

    def test_msr_control(self, asm):
        assert len(asm.asm("msr CONTROL, r0")) == 4

    def test_mrs_basepri_max(self, asm):
        assert len(asm.asm("mrs r0, BASEPRI_MAX")) == 4


# ---------------------------------------------------------------------------
# FPv5 instructions (also missing from keystone)
# ---------------------------------------------------------------------------

class TestFPv5:
    def test_vfma(self, asm):
        assert asm.asm("vfma.f32 s0, s1, s2") == bytes.fromhex("a0ee810a")

    def test_vfms(self, asm):
        assert len(asm.asm("vfms.f32 s0, s1, s2")) == 4

    def test_vfnma(self, asm):
        assert len(asm.asm("vfnma.f32 s0, s1, s2")) == 4

    def test_vmov_f32_imm(self, asm):
        assert len(asm.asm("vmov.f32 s0, #1.0")) == 4

    def test_vldr_vstr(self, asm):
        assert len(asm.asm("vldr s0, [r0, #0]\nvstr s0, [r1, #4]")) == 8

    def test_vcvt_f64_s32(self, asm):
        assert len(asm.asm("vcvt.f64.s32 d0, s0")) == 4

    def test_vcvt_f32_f64(self, asm):
        assert len(asm.asm("vcvt.f32.f64 s0, d0")) == 4

    def test_fpu_full_register_file(self, asm):
        """s0-s31 and d0-d15."""
        assert len(asm.asm("vmov s31, r0")) == 4
        assert len(asm.asm("vmov.f64 d15, d0")) == 4

    def test_vpush_vpop(self, asm):
        code = asm.asm("vpush {s0-s3}\nvpop {s0-s3}")
        assert len(code) == 8

    def test_vmrs_fpscr(self, asm):
        assert len(asm.asm("vmrs APSR_nzcv, FPSCR")) == 4


# ---------------------------------------------------------------------------
# Labels and branches
# ---------------------------------------------------------------------------

class TestLabels:
    def test_label_forward(self, asm):
        code = asm.asm("b target\nnop\ntarget:\nnop")
        assert len(code) >= 4

    def test_label_backward(self, asm):
        code = asm.asm("loop:\nnop\nb loop")
        assert len(code) >= 4

    def test_local_labels(self, asm):
        code = asm.asm("1:\nnop\nb 1b")
        assert len(code) >= 4

    def test_branch_self(self, asm):
        code = asm.asm("here:\nb here")
        assert len(code) in (2, 4)


# ---------------------------------------------------------------------------
# Literal pools
# ---------------------------------------------------------------------------

class TestLiteralPools:
    def test_ldr_equals(self, asm):
        """ldr r0, =0xDEADBEEF — constant pool expansion."""
        code = asm.asm("ldr r0, =0xDEADBEEF\n.ltorg")
        assert len(code) >= 4
        assert b"\xef\xbe\xad\xde" in code

    def test_ldr_equals_small(self, asm):
        """Small constant may use mov instead of literal pool."""
        code = asm.asm("ldr r0, =0")
        assert len(code) in (2, 4)


# ---------------------------------------------------------------------------
# Width specifiers (.n / .w)
# ---------------------------------------------------------------------------

class TestWidthSpecifiers:
    def test_narrow_nop(self, asm):
        assert asm.asm("nop.n") == b"\x00\xbf"  # 16-bit

    def test_wide_nop(self, asm):
        code = asm.asm("nop.w")
        assert len(code) == 4  # 32-bit

    def test_narrow_branch(self, asm):
        code = asm.asm("here_n:\nb.n here_n")
        assert len(code) == 2

    def test_wide_branch(self, asm):
        code = asm.asm("here_w:\nb.w here_w")
        assert len(code) == 4


# ---------------------------------------------------------------------------
# IT blocks
# ---------------------------------------------------------------------------

class TestITBlocks:
    def test_it_eq(self, asm):
        code = asm.asm("it eq\nmoveq r0, #1")
        assert len(code) == 4  # IT (2) + mov (2)

    def test_ite(self, asm):
        code = asm.asm("ite eq\nmoveq r0, #1\nmovne r0, #0")
        assert len(code) == 6  # IT (2) + mov (2) + mov (2)

    def test_ittt(self, asm):
        code = asm.asm("ittt eq\nmoveq r0, #1\nmoveq r1, #2\nmoveq r2, #3")
        assert len(code) == 8


# ---------------------------------------------------------------------------
# Addressing modes
# ---------------------------------------------------------------------------

class TestAddressingModes:
    def test_immediate_offset(self, asm):
        assert len(asm.asm("ldr r0, [r1, #4]")) in (2, 4)

    def test_post_increment(self, asm):
        assert len(asm.asm("ldr r0, [r1], #4")) == 4

    def test_shifted_register(self, asm):
        assert len(asm.asm("ldr.w r0, [r1, r2, lsl #3]")) == 4

    def test_byte_halfword_loads(self, asm):
        for mnemonic in ["ldrb r0, [r1]", "ldrh r0, [r1]",
                         "ldrsb r0, [r1, r2]", "ldrsh r0, [r1, r2]"]:
            code = asm.asm(mnemonic)
            assert len(code) in (2, 4), f"failed: {mnemonic}"


# ---------------------------------------------------------------------------
# DSP / saturating instructions
# ---------------------------------------------------------------------------

class TestDSP:
    def test_usat(self, asm):
        assert len(asm.asm("usat r3, #0xf, r3")) == 4

    def test_ssat(self, asm):
        assert len(asm.asm("ssat r0, #16, r1")) == 4

    def test_smull(self, asm):
        assert len(asm.asm("smull r0, r1, r2, r3")) == 4

    def test_umull(self, asm):
        assert len(asm.asm("umull r0, r1, r2, r3")) == 4

    def test_qadd(self, asm):
        assert len(asm.asm("qadd r0, r1, r2")) == 4

    def test_qsub(self, asm):
        assert len(asm.asm("qsub r0, r1, r2")) == 4


# ---------------------------------------------------------------------------
# Thumb-2 modified immediates
# ---------------------------------------------------------------------------

class TestModifiedImmediates:
    def test_repeated_byte_pattern(self, asm):
        """mov.w with 0x01010101 repeated-byte encoding."""
        assert len(asm.asm("mov.w r0, #0x01010101")) == 4

    def test_movw_movt_pair(self, asm):
        code = asm.asm("movw r0, #0x1234; movt r0, #0x5678")
        assert len(code) == 8

    def test_shifted_immediate(self, asm):
        assert len(asm.asm("orr r0, r0, #0x10001")) == 4


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_cortex_m7_sp(self):
        a = Assembler.cortex_m7_sp()
        assert a.cpu == "cortex-m7"
        assert a.asm("vfma.f32 s0, s1, s2") == bytes.fromhex("a0ee810a")
        with pytest.raises(AsmError):
            a.asm("vmov.f64 d0, d1")

    def test_cortex_m7_dp(self):
        a = Assembler.cortex_m7_dp()
        assert a.cpu == "cortex-m7"
        assert len(a.asm("vmov.f64 d0, d1")) == 4
        assert a.asm("vfma.f32 s0, s1, s2") == bytes.fromhex("a0ee810a")

    def test_cortex_m4(self):
        a = Assembler.cortex_m4()
        assert a.cpu == "cortex-m4"
        assert len(a.asm("vadd.f32 s0, s1, s2")) == 4

    def test_cortex_m33(self):
        a = Assembler.cortex_m33()
        assert a.cpu == "cortex-m33"
        assert len(a.asm("sg")) == 4
        assert len(a.asm("tt r0, r1")) == 4

    def test_cortex_m0_no_thumb2(self):
        a = Assembler.cortex_m0()
        assert a.cpu == "cortex-m0"
        assert a.asm("nop") == b"\x00\xbf"
        with pytest.raises(AsmError):
            a.asm("movw r0, #0x1234")

    def test_bare_assembler_raises(self):
        with pytest.raises(TypeError, match="requires a target"):
            Assembler()


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_with_cpu(self):
        a = Assembler.cortex_m7_dp()
        assert repr(a) == "Assembler(thumbv7em-none-eabi, cortex-m7)"

    def test_repr_no_cpu(self):
        a = Assembler.x86_64()
        assert repr(a) == "Assembler(x86_64)"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_hit(self):
        a = Assembler.cortex_m7_dp()
        _ = a.asm("nop")
        assert a.cache_size >= 1
        _ = a.asm("nop")
        assert a.cache_size >= 1

    def test_cache_clear(self):
        a = Assembler.cortex_m7_dp()
        _ = a.asm("nop")
        a.cache_clear()
        assert a.cache_size == 0

    def test_cached_result_identical(self):
        a = Assembler.cortex_m7_dp()
        first = a.asm("mov r0, #42")
        second = a.asm("mov r0, #42")
        assert first is second


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_instruction(self, asm):
        with pytest.raises(AsmError):
            asm.asm("not_a_real_instruction r0, r1")

    def test_bad_register(self, asm):
        with pytest.raises(AsmError):
            asm.asm("mov r99, #0")

    def test_bad_llvm_path(self):
        with pytest.raises(FileNotFoundError):
            Assembler(triple="x86_64", llvm_mc="/nonexistent/llvm-mc")

    def test_empty_input(self, asm):
        code = asm.asm("")
        assert code == b""

    def test_error_line_numbers(self):
        """Error line numbers should map to user source, not internal preamble."""
        a = Assembler.cortex_m7_dp()
        with pytest.raises(AsmError, match="line 2:"):
            a.asm("nop\nbad_instruction_here")


# ---------------------------------------------------------------------------
# Semicolon handling
# ---------------------------------------------------------------------------

class TestSemicolons:
    def test_bare_semicolons(self, asm):
        """Bare semicolons split instructions."""
        code = asm.asm("mov r0, #0; bx lr")
        assert len(code) >= 4

    def test_quoted_semicolons_preserved(self, asm):
        """Semicolons inside .ascii strings are not split."""
        code = asm.asm('.ascii "hello;world"\n.align 2')
        assert b"hello;world" in code

    def test_comment_semicolons_preserved(self, asm):
        """Semicolons after @ comment markers are not split."""
        code = asm.asm("nop @ comment; not a split")
        assert code == b"\x00\xbf"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_asm(self):
        """Multiple threads can call asm() without corruption."""
        import concurrent.futures

        a = Assembler.cortex_m7_dp()
        instructions = [f"mov r{i % 8}, #{i}" for i in range(100)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(a.asm, instr) for instr in instructions]
            results = [f.result() for f in futures]

        # All should produce non-empty bytes
        assert all(len(r) > 0 for r in results)
        # Cache should have entries
        assert a.cache_size > 0
