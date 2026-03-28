# TODO
## Architecture & Refactoring

- [ ] **Split Backends**: Move `subprocess` backend logic out of `assembler.py` into `subprocess_backend.py`.
- [ ] **Direct Emission**: Investigate bypassing ELF generation to emit raw machine code directly from the MC layer for better performance.
- [ ] **Extensibility**: Ensure the `Assembler` factory can easily accommodate future backends (e.g., Keystone/Capstone for round-tripping).
- [ ] **Architecture Inference**: Improve `_arch_for_triple` to potentially handle unmapped triples by attempting to resolve symbols for common prefixes automatically.

## API Enhancements

- [ ] **Instruction Count**: Add `asm_info()` or update `asm()` to return a result object with `.bytes` and `.count` (instruction count).
- [ ] **Big-Endian Support**: Update ELF parser to detect and handle Big-Endian objects (required for MIPS/PPC).
- [ ] **Symbol Injection**: Allow users to provide a mapping of external symbols to addresses (similar to Nyxstone).
- [ ] **Address-Aware Assembly**: Support base address (PC) for position-relative instructions.
- [ ] **Round-trip validation**: Add an optional validation step that disassembles the output (via Capstone) to verify it matches the input mnemonic.
- [ ] **`asm_each()`**: Return per-instruction boundaries (offset, size, bytes) for breakpoint placement and fine-grained binary analysis.

## Profiles & Architectures
...
- [ ] **Dockerfile**: Finalize whether to keep the `Dockerfile` in the repo for consistent multi-distro testing.
- [ ] **Logging**: Add a `logging` logger to record LLVM discovery steps and backend selection events.

## Documentation
- [ ] **Cortex-M Expansion**: `Assembler.cortex_m23()`, `Assembler.cortex_m55()`.
- [ ] **Cortex-A Profiles**: `Assembler.armv7a()`, `Assembler.armv8a()`.

## Platform & Discovery

- [ ] **Windows Validation**: Verify the new `win32` discovery logic on a real Windows environment.
- [ ] **Versioned Linux Discovery**: Refine the `glob` logic to prioritize the highest available version if multiple `libLLVM-N.so` are found.
- [ ] **Dockerfile**: Finalize whether to keep the `Dockerfile` in the repo for consistent multi-distro testing.

## Documentation

- [ ] **README Update**:
    - [ ] Document the `--backend` flag for `pytest`.
    - [ ] Add technical details for `register_arch_mapping`.
    - [ ] Add technical details for `set_libllvm_path`.
    - [ ] Update Windows support status.
- [ ] **Testing Guide**: Add a dedicated section on how to run cross-validation tests (GAS, Keystone, Nyxstone).

## Done

- [x] **Specific Error Reporting**: C API failures now report the exact LLVM function name and diagnostic string.
- [x] **Unified Exception Hierarchy**: `AsmError` and `UnsupportedArchitectureError` are consolidated and exported.
- [x] **Expanded Arch Mapping**: Support for WebAssembly, PowerPC, Sparc, SystemZ, Mips, LoongArch.
- [x] **Windows Support**: Implementation of dynamic discovery for `LLVM-C.dll` and `llvm-mc.exe`.
- [x] **Linux Discovery**: Robust search for versioned `libLLVM.so` using `glob` and multiarch paths.
- [x] **True Concurrency**: Thread-Local Storage (TLS) for C API backends allows parallel assembly.
- [x] **Resource Management**: Proactive finalization of LLVM objects and safe interpreter teardown.
- [x] **Technical Documentation**: Comprehensive `TECHNICAL.md` and ecosystem `COMPARISON.md`.
