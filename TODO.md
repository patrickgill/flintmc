# TODO

## Architecture support

- [x] x86 (32-bit) — `Assembler.i686()`, Intel syntax default
- [x] x86_64 — `Assembler.x86_64()`, Intel syntax default
- [x] AArch64 — `Assembler.aarch64()`
- [x] 64-bit ELF support in `.text` extractor
- [ ] RISC-V profiles
- [ ] MIPS profiles

## In-process LLVM C API backend (researched, not yet integrated)

Proof of concept in `research/llvm-capi-backend.py`. Uses only the stable LLVM C API — no C++ shim needed.

### Pipeline

1. `LLVMInitialize{Arch}*()` — one-time target init per architecture
2. `LLVMGetTargetFromTriple()` + `LLVMCreateTargetMachine()` — reusable across calls
3. Per assembly call:
   - `LLVMModuleCreateWithName()` — throwaway module
   - `LLVMSetModuleInlineAsm2()` — inject assembly source
   - `LLVMTargetMachineEmitToMemoryBuffer()` — emit ELF object in-memory
   - Extract `.text` from ELF (existing Python code)
   - `LLVMDisposeMemoryBuffer()` + `LLVMDisposeModule()`

### Performance (Apple M4 Max, LLVM 22.1, 100 unique instructions)

| Backend    | Per-call | Total  |
|------------|----------|--------|
| C API      | 0.2ms    | 0.015s |
| Subprocess | 10.1ms   | 1.010s |
| **Speedup** | **65x** |        |

### Known limitations vs subprocess (llvm-mc)

- **Thumb relaxation**: `mov r0, #42` emits wide `mov.w` (4B) via C API vs narrow `movs` (2B) via llvm-mc. The standalone llvm-mc pipeline runs `MCAssembler::relaxInstruction()`; the module inline asm path through `AsmPrinter` skips this. Workaround: use `movs` explicitly for narrow encoding.
- **Error diagnostics**: assembly errors go through LLVM's diagnostic handler, not stderr. Need `LLVMContextSetDiagnosticHandler` to capture them programmatically. Without it, errors print to stderr and the emit call returns failure with a generic message.
- **Module overhead**: each call creates/destroys an `LLVMModule` (~0.01ms, trivial but involves malloc/free churn).

### Critical ctypes gotcha

ALL function `.argtypes` MUST be declared before the first call. Without argtypes, ctypes assumes `c_int` (32-bit) for all arguments. On ARM64 macOS (any LP64 platform), this silently truncates 64-bit pointers to 32 bits. LLVM functions receive garbage pointers and segfault — typically `LLVMDisposeTargetMachine` or `LLVMCreateTargetDataLayout`.

Crash signature: `EXC_BAD_ACCESS at 0xfffffffff10c4600` — the `0xffffffff` prefix is the sign-extended upper 32 bits of a truncated pointer.

### Integration plan

- [ ] Wrap as `LlvmCApiBackend` alongside existing subprocess backend
- [ ] `Assembler` auto-detects: C API if `libLLVM` found, subprocess fallback
- [ ] Error capture via `LLVMContextSetDiagnosticHandler`
- [ ] Thread safety: serialize module creation behind a lock (TargetMachine is read-only after creation, safe to share)

## Features

- [ ] Round-trip validation: assemble then disassemble with Capstone, verify mnemonic matches
- [x] ~~Persistent subprocess mode~~ — superseded by C API backend (65x faster)
- [ ] `asm_each()` returning per-instruction boundaries (offset, size, bytes) for breakpoint/end-address calculation

## Profiles

- [ ] Cortex-M23 (ARMv8-M Baseline)
- [ ] Cortex-M55 (ARMv8.1-M Mainline + MVE)
- [ ] Cortex-A (ARMv7-A, ARMv8-A) for non-M use cases
