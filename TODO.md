# TODO

add testing guidance to the readme if it's not there, or update it
add how to for the different backend testing


# pytest
[x] can we pass options to pytest instead of hardcoding?
[x] find out how to run the tests on the llvm-mc subprocess backend too
[x] tests should say which backend is being used (either C API or llvm-mc subprocess)
it says other stuff at the top of the test so why not this
there should be some structured way to do this

think if there's other things to mention too


# Dockerfile
Keep Dockerfile or gitignore?

things wanted are
Test different Ubuntu versions
Test LLVM installation
make sure the tests work on each
Test different Python versions

If we're testing LLVM 15-18 the Nyxstone tests can be done


## Done

- [x] x86 (32-bit) — `Assembler.i686()`, Intel syntax default
- [x] x86_64 — `Assembler.x86_64()`, Intel syntax default
- [x] AArch64 — `Assembler.aarch64()`
- [x] 64-bit ELF support in `.text` extractor
- [x] LLVM C API backend — auto-detected as default
- [x] `LLVMContextSetDiagnosticHandler` error interception (no more `exit()`)
- [x] Thread-safe C API (lock-serialized module creation/emit)
- [x] Subprocess fallback when libLLVM unavailable
- [x] CI testing for both backends (C API and subprocess)
- [x] Visible backend status in pytest header

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

- **LLVM minimum version**: 7.0+ required for `LLVMSetModuleInlineAsm2`. Other functions we use date back to LLVM 3.5–3.9. Tested up to LLVM 22.x.
- **ctypes argtypes**: ALL function `.argtypes` MUST be declared before calling. Without them, ctypes truncates 64-bit pointers on ARM64 macOS → SIGSEGV. Crash signature: `EXC_BAD_ACCESS at 0xfffffffff...`
- **LLVM Context Isolation**: The C API backend reuses a single `LLVMModuleRef` and `LLVMTargetDataRef` for maximum performance. This makes the backend instance strictly serial and dependent on the `threading.Lock` in `Assembler` to prevent memory corruption.
