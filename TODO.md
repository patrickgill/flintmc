# TODO

## Done

- [x] x86 (32-bit) — `Assembler.i686()`, Intel syntax default
- [x] x86_64 — `Assembler.x86_64()`, Intel syntax default
- [x] AArch64 — `Assembler.aarch64()`
- [x] 64-bit ELF support in `.text` extractor
- [x] LLVM C API backend — 65x faster, auto-detected as default
- [x] `LLVMContextSetDiagnosticHandler` error interception (no more `exit()`)
- [x] Thread-safe C API (lock-serialized module creation/emit)
- [x] Subprocess fallback when libLLVM unavailable
- [x] ~~Persistent subprocess mode~~ — superseded by C API backend

## Open

### Features

- [ ] Round-trip validation: assemble then disassemble with Capstone, verify mnemonic matches
- [ ] `asm_each()` returning per-instruction boundaries (offset, size, bytes) for breakpoint/end-address calculation

### Profiles

- [ ] RISC-V profiles
- [ ] MIPS profiles
- [ ] Cortex-M23 (ARMv8-M Baseline)
- [ ] Cortex-M55 (ARMv8.1-M Mainline + MVE)
- [ ] Cortex-A (ARMv7-A, ARMv8-A) for non-M use cases

### Known limitations

- **Thumb relaxation**: C API path emits `mov r0, #42` as wide `mov.w` (4B) instead of narrow `movs` (2B). Use `movs` explicitly for narrow encoding. The standalone llvm-mc pipeline runs `MCAssembler::relaxInstruction()` which the module inline asm path skips.
- **ctypes argtypes**: ALL function `.argtypes` MUST be declared before calling. Without them, ctypes truncates 64-bit pointers on ARM64 macOS → SIGSEGV. Crash signature: `EXC_BAD_ACCESS at 0xfffffffff...`
