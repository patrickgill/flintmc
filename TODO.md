# TODO

verify=True tests:
manually do an asm and disasm() and compare those to the a regular asm() only that has verify=True


## Next up

- [ ] **Uniform symbol branches**: make branches to `symbols=` values absolute on every arch (define them relative to a start-of-block label?), without breaking their use as immediates.
- [ ] **Cortex-M expansion**: `Assembler.cortex_m23()`, `Assembler.cortex_m55()`.
- [ ] **Cortex-A profiles**: `Assembler.armv7a()`, `Assembler.armv8a()`.
- [ ] **Add verify=True to applicable tests**: Opt in to round-trip verification across the test suite.

## Backlog

- [ ] **Direct emission**: Investigate bypassing ELF generation to emit raw machine code directly from the MC layer for better performance.
- [ ] **Extensibility**: Ensure the `Assembler` factory can easily accommodate future backends (e.g., Keystone/Capstone for round-tripping).
- [ ] **Architecture inference**: Improve `_arch_for_triple` to handle unmapped triples by attempting to resolve symbols for common prefixes automatically.
- [ ] **Windows validation**: Verify discovery logic on a real Windows environment.
- [ ] **Versioned Linux discovery**: Prioritize the highest available version if multiple `libLLVM-N.so` are found.
- [ ] **Dockerfile**: Finalize whether to keep the `Dockerfile` in the repo for consistent multi-distro testing.
- [ ] **Logging**: Add a `logging` logger to record LLVM discovery steps and backend selection events.

## Done

- [x] **ELF triple normalization**: Apple/Windows triples assembled as ELF (`_elf_triple`). Fixes Mach-O local-label branches and the COFF `.set`+`jmp` crash.
- [x] **Disassembler**: `Disassembler` class with `LLVMDisasmInstruction` C API.
- [x] **Symbol injection**: `symbols={}` parameter with `.set` directives and name validation.
- [x] **ELF relocation patching**: `R_X86_64_PC32`, `R_X86_64_PLT32` (ELF64), `R_386_PC32` (ELF32).
- [x] **Base address**: `address=` any 64-bit value (relocation base + low-16-bit `.org`).
- [x] **Relocation patcher**: undefined symbols and unknown relocations raise; AArch64 adrp/lo12/branch, absolute data relocations applied.
- [x] **Per-instruction info**: `asm_each()` with `InstructionInfo`.
- [x] **Round-trip verification**: `verify=True`.
- [x] **Mach-O support**: 32/64-bit `__text` extraction.
- [x] **COFF support**: `.text` extraction for Windows triples.
- [x] **Big-endian ELF**: Reads `EI_DATA` for endianness.
- [x] **Context manager**: `close()` + `with` statement on `Assembler` and `Disassembler`.
- [x] **Extensible preambles**: `register_default_preamble()`.
- [x] **New profiles**: riscv64/32, avr, bpf, msp430, loongarch64.
- [x] **arm64 prefix fix**: Sorted by prefix length so `arm64` matches before `arm`.
- [x] **armv*a preamble fix**: `.arm` not `.thumb` for Cortex-A triples.
- [x] **Semicolon splitter fixes**: Single quotes, `#` immediate heuristic.
- [x] **Linux discovery fix**: Tighter glob, catches RuntimeError from component libraries.
- [x] **Backend cache invalidation**: `set_libllvm_path()` resets `_resolved_default`.
- [x] **Refactored common.py**: Split into `errors.py`, `objfile.py`, `common.py`.
- [x] **Specific error reporting**: C API failures report exact LLVM function name and diagnostic.
- [x] **Unified exception hierarchy**: `AsmError` and `UnsupportedArchitectureError`.
- [x] **Expanded arch mapping**: PowerPC, Sparc, SystemZ, Mips, LoongArch, AVR, BPF, MSP430.
- [x] **Windows support**: Discovery for `LLVM-C.dll` and `llvm-mc.exe`.
- [x] **Linux discovery**: Versioned `libLLVM.so` using `glob` and multiarch paths.
- [x] **True concurrency**: Thread-Local Storage for C API and disassembler backends.
- [x] **Resource management**: `weakref.finalize` for LLVM objects.
- [x] **Technical documentation**: `TECHNICAL.md` and `COMPARISON.md`.
- [x] **5166 tests**: Concurrency stress, adversarial symbol injection, multi-arch, round-trip.
