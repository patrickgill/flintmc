# TODO

## Architecture support

- [ ] x86 (32-bit) — LLVM triple `i686-unknown-linux-gnu` or similar
- [ ] x86_64 — LLVM triple `x86_64-unknown-linux-gnu`
- [ ] AArch64 — LLVM triple `aarch64-none-elf`

These should be straightforward — llvm-mc already handles all of them. The `Assembler` class just needs appropriate defaults and profiles per arch. The ELF parser may need 64-bit ELF support for x86_64/AArch64.

## Features

- [ ] Round-trip validation: assemble then disassemble with Capstone, verify mnemonic matches
- [ ] Persistent subprocess mode (keep llvm-mc alive with stdin/stdout pipe) for batch assembly perf
- [ ] `asm_each()` returning per-instruction boundaries (offset, size, bytes) for breakpoint/end-address calculation

## Profiles

- [ ] Cortex-M23 (ARMv8-M Baseline)
- [ ] Cortex-M55 (ARMv8.1-M Mainline + MVE)
- [ ] Cortex-A (ARMv7-A, ARMv8-A) for non-M use cases
