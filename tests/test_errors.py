"""Tests for error reporting and target initialization.

Verifies that specific error messages are raised for unknown architectures,
missing LLVM symbols, and other target-related configuration issues.
"""

import pytest
import flintmc.llvm_capi
from flintmc import Assembler, AsmError

@pytest.fixture
def clean_arch_mapping():
    """Temporarily modify the architecture mapping and restore it after the test."""
    original = dict(flintmc.llvm_capi._TRIPLE_TO_ARCH)
    yield
    flintmc.llvm_capi._TRIPLE_TO_ARCH = original

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_unsupported_triple_error():
    """Verify error for a triple that cannot be mapped to an LLVM architecture."""
    triple = "unknown-arch-none-eabi"
    # Force CAPI to ensure we trigger the llvm_capi logic
    with pytest.raises(AsmError) as excinfo:
        asm = Assembler(triple=triple, backend="capi")
        asm("nop")
    
    assert "Unsupported or unknown architecture for triple: unknown-arch-none-eabi" in str(excinfo.value)

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_missing_symbol_error(clean_arch_mapping):
    """Verify error when an LLVM architecture symbol is missing from the library."""
    # Add a fake arch to _TRIPLE_TO_ARCH that will definitely not have symbols
    flintmc.llvm_capi._TRIPLE_TO_ARCH["fake"] = "FakeArch"
    triple = "fake-triple"
    
    with pytest.raises(AsmError) as excinfo:
        asm = Assembler(triple=triple, backend="capi")
        asm("nop")
    
    assert "LLVM symbol 'LLVMInitializeFakeArchTargetInfo' not found" in str(excinfo.value)
    assert "Is FakeArch support enabled in your LLVM build?" in str(excinfo.value)
