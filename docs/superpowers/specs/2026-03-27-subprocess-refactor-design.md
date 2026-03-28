# Design Spec: Subprocess Backend Refactor and Project Reorganization

**Date:** 2026-03-27
**Topic:** Refactor subprocess backend and consolidate common utilities.
**Version:** 0.2.0

## 1. Overview
Currently, `flintmc/assembler.py` contains both the high-level API and the low-level `llvm-mc` subprocess logic, along with many utility functions shared across backends. This refactor extracts the subprocess logic into `flintmc/llvm_subprocess.py` and moves shared utilities and exceptions into `flintmc/common.py`.

## 2. Goals
- Extract `llvm-mc` subprocess logic into its own module.
- Centralize exceptions and utilities to avoid duplication and circular imports.
- Maintain the current API for `Assembler` while simplifying its implementation.
- Update project version to `0.2.0`.

## 3. Component Design

### 3.1 `flintmc/common.py`
This file will host shared logic that does not depend on specific backends:
- **Exceptions**: `AsmError`, `UnsupportedArchitectureError`.
- **Global Config**:
    - `set_llvm_mc_path(path: str)`: Set a global path to the `llvm-mc` binary.
- **Discovery**: `find_llvm_mc()` (updated to check for globally set path first).
- **ELF Extraction**: `_extract_text` and its internal helpers (`_extract_text_elf32`, `_extract_text_elf64`, `_find_text`).
- **Assembly Helpers**: `_split_semicolons`, `_fix_error`, `_default_preamble`.

### 3.2 `flintmc/llvm_subprocess.py`
This file will contain the `LlvmSubprocessBackend` class:
- **Interface**:
    - `__init__(self, triple: str, cpu: str, features: str, llvm_mc: str)`
    - `asm(self, source: str, preamble_lines: int) -> bytes`
- **Implementation**:
    - Uses `subprocess.run` to call `llvm-mc`.
    - Returns the raw ELF bytes produced by `llvm-mc`.
    - Raises `AsmError` with adjusted line numbers (using `_fix_error`) on failure.

### 3.3 `flintmc/llvm_capi.py`
Refactored to:
- Import `AsmError` and `UnsupportedArchitectureError` from `common.py`.
- Remove its own definitions of these exceptions.

### 3.4 `flintmc/assembler.py`
The main entry point:
- Removes ~300 lines of moved logic.
- Maintains `Assembler` class, profile methods (`cortex_m7_dp`, `x86_64`, etc.), and caching.
- Orchestrates:
    1. Pre-processes source (semicolons, preamble).
    2. Delegates to backend (C API or Subprocess) which returns ELF bytes.
    3. Post-processes ELF via `common._extract_text`.

## 4. Public API & Versioning
- **Re-exports**: To maintain backward compatibility and a clean "single import" experience:
    - `flintmc/common.py`: Definitive home for `AsmError`, `UnsupportedArchitectureError`, `find_llvm_mc`, and `set_llvm_mc_path`.
    - `flintmc/assembler.py`: Will import and re-export `AsmError`, `UnsupportedArchitectureError`, `find_llvm_mc`, and `set_llvm_mc_path`.
    - `flintmc/__init__.py`: Will continue to export these symbols from their new internal locations.
- `flintmc/__init__.py`: Update `__version__` to `0.2.0`.
- `pyproject.toml`: Update `version` to `0.2.0`.

## 5. Verification Plan
- Run existing tests: `pytest tests/`
- Verify both backends work: `pytest --backend=capi` and `pytest --backend=subprocess`.
- Ensure `AsmError` can still be caught via `from flintmc.assembler import AsmError` and `from flintmc import AsmError`.
