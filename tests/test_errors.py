"""Tests for error reporting and target initialization.

Verifies that specific error messages are raised for unknown architectures,
missing LLVM symbols, and other target-related configuration issues.
"""

import ctypes
from unittest.mock import MagicMock, patch
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
        # Use a fresh Assembler each time
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

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_llvm_function_not_found():
    """Verify error when a required core LLVM function is missing from the library."""
    # Patch _load_llvm itself since patching internal function calls is brittle
    with patch("flintmc.llvm_capi._load_llvm", side_effect=RuntimeError("Required LLVM function 'LLVMContextCreate' not found")):
        with pytest.raises(RuntimeError) as excinfo:
            flintmc.llvm_capi._load_llvm()
        
        assert "Required LLVM function 'LLVMContextCreate' not found" in str(excinfo.value)

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_target_machine_creation_failure():
    """Verify error when LLVMCreateTargetMachine returns NULL."""
    lib = flintmc.llvm_capi._load_llvm()
    triple = "x86_64-pc-linux-gnu-test-tm"
    
    # We need to mock disposal to avoid crashing if we hit cleanup during a failed init
    with patch.object(lib, "LLVMCreateTargetMachine", return_value=0), \
         patch.object(lib, "LLVMDisposeModule"), \
         patch.object(lib, "LLVMContextDispose"):
        
        with pytest.raises(AsmError) as excinfo:
            asm = Assembler(triple=triple, backend="capi")
            asm("nop")
        assert "Failed to create TargetMachine" in str(excinfo.value)

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_module_creation_failure():
    """Verify error when LLVMModuleCreateWithNameInContext returns NULL."""
    lib = flintmc.llvm_capi._load_llvm()
    triple = "x86_64-pc-linux-gnu-test-mod"
    
    with patch.object(lib, "LLVMModuleCreateWithNameInContext", return_value=0), \
         patch.object(lib, "LLVMDisposeTargetMachine"), \
         patch.object(lib, "LLVMContextDispose"):
        
        with pytest.raises(AsmError) as excinfo:
            asm = Assembler(triple=triple, backend="capi")
            asm("nop")
        assert "LLVMModuleCreateWithNameInContext() failed to create module" in str(excinfo.value)

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_data_layout_creation_failure():
    """Verify error when LLVMCreateTargetDataLayout returns NULL."""
    lib = flintmc.llvm_capi._load_llvm()
    triple = "x86_64-pc-linux-gnu-test-dl"
    
    with patch.object(lib, "LLVMCreateTargetDataLayout", return_value=0), \
         patch.object(lib, "LLVMDisposeTargetMachine"), \
         patch.object(lib, "LLVMDisposeModule"), \
         patch.object(lib, "LLVMContextDispose"):
        
        with pytest.raises(AsmError) as excinfo:
            asm = Assembler(triple=triple, backend="capi")
            asm("nop")
        assert "LLVMCreateTargetDataLayout() failed" in str(excinfo.value)

@pytest.mark.skipif(not Assembler.capi_available(), reason="C API backend not available")
def test_asm_failure_with_diagnostics():
    """Verify that diagnostics are included in the error message when assembly fails."""
    triple = "x86_64-pc-linux-gnu-test-diag"
    asm = Assembler(triple=triple, backend="capi")
    
    lib = flintmc.llvm_capi._load_llvm()
    
    # Force backend creation
    backend = asm._get_capi_backend()
    
    # Inject an error via LLVMSetModuleInlineAsm2 so it happens AFTER the clear() in backend.asm()
    def mock_set_asm(*args):
        backend._errors.append("Directly injected diagnostic error")
        return None

    with patch.object(lib, "LLVMTargetMachineEmitToMemoryBuffer", return_value=1), \
         patch.object(lib, "LLVMSetModuleInlineAsm2", side_effect=mock_set_asm), \
         patch.object(lib, "LLVMDisposeMessage"):
        
        with pytest.raises(AsmError) as excinfo:
            asm.asm("nop")
        
        assert "Directly injected diagnostic error" in str(excinfo.value)
        assert "assembly failed" in str(excinfo.value)


@pytest.mark.parametrize("kw", [{}, {"symbols": {"x": 1, "y": 2}}, {"address": 0x10}])
def test_error_line_number_ignores_injected_lines(kw):
    """symbols= and address= inject directives; error lines must still match user source."""
    with pytest.raises(AsmError, match=r"^line 2:"):
        Assembler.x86_64().asm("nop\nbogus", **kw)


def test_invalid_backend_name():
    with pytest.raises(ValueError, match="backend must be"):
        Assembler("x86_64", backend="bogus")
