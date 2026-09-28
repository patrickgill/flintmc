# Changelog

## Unreleased

- `Disassembler` has every `Assembler` profile (`cortex_m0/m4/m7_sp/m33`, `armv6m/7m/8m`, `loongarch64`, `avr`, `bpf`, `msp430`, `i686` alias). Both read one shared table, so settings can't drift.
- Profile methods accept `cpu=`/`features=` overrides (previously a `TypeError`).
- `asm_each()` accepts `address=` and `symbols=`, like `asm()`.
- `address=` accepts any 64-bit value (was capped at 1 MiB). STM32-style `0x08000000` now takes milliseconds instead of ~0.6 s and 400 MB.
- Undefined symbols raise `AsmError` instead of assembling to garbage (`call prinft` was a call to 0; AArch64 `bl prinft` a branch to itself).
- AArch64 `adrp`/`:lo12:` pairs and branches/`adr`/`ldr` to `.global` labels are resolved (were left zeroed). Absolute references to labels (`movabs $label`, AArch64/RISC-V `ldr =label`) hold the label's address (were 0). Other relocations LLVM leaves in `.text` raise instead of shipping an unpatched field.
- Docs: branches to `symbols=` values are absolute on x86 but a displacement from the instruction on ARM/AArch64/RISC-V (previously documented as "offset from block start").
- GNU binutils backend for targets LLVM lacks, with `v850()`, `v850es()` and `v850e1()` profiles on `Assembler` and `Disassembler`. It is picked automatically when LLVM has no target for the triple and `<arch>-elf-as` is available (`$BINUTILS_PATH/bin` or `PATH`).

## 0.3.0

### New features
- **Disassembler**: `Disassembler` class using LLVM's `LLVMDisasmInstruction` C API. Profiles for x86_64, x86_32, AArch64, ARM Thumb, RISC-V. Thread-safe via TLS. C API only (no subprocess fallback).
- **Symbol injection**: `symbols={"handler": 0x8000}` parameter on `asm()`. Injects `.set` directives with name validation to prevent injection. Branch targets on x86 are resolved via ELF relocation patching.
- **Base address**: `address=0x1000` parameter on `asm()`. Uses `.org` internally, capped at 1 MiB.
- **Per-instruction info**: `asm_each()` returns `InstructionInfo(offset, size, code, source)` for each instruction. Uses cumulative assembly for correct label/forward-reference handling.
- **Round-trip verification**: `verify=True` disassembles output and re-assembles it, raising `AsmError` on byte mismatch.
- **Context manager**: `Assembler` and `Disassembler` support `with` statements and `close()`.
- **Extensible preambles**: `register_default_preamble(prefix, preamble)` for custom architectures.

### Object format support
- **Mach-O**: 32-bit and 64-bit `__text` extraction. Apple triples (`arm64-apple-macos`) now work.
- **COFF**: `.text` extraction for Windows triples (`x86_64-pc-windows-msvc`).
- **Big-endian ELF**: Reads `EI_DATA` byte to select endianness. Required for PowerPC, Sparc, SystemZ, MIPS targets.
- **ELF relocation patching**: Applies `R_X86_64_PC32`/`R_X86_64_PLT32` (ELF64) and `R_386_PC32` (ELF32) relocations for branch-to-symbol resolution.

### Architecture profiles
- New profiles: `riscv64()`, `riscv32()`, `avr()`, `bpf()`, `msp430()`, `loongarch64()`.
- Removed `wasm32`/`wasm64` arch mappings (LLVM emits wasm binary, not ELF/Mach-O/COFF).

### Fixes
- `arm64` triple prefix now correctly resolves to AArch64 (not ARM).
- `armv7a`/`armv8a` triples get `.arm` preamble (not `.thumb`).
- `_split_semicolons`: tracks single quotes, treats `#42` and `#(expr)` as immediates (not comments).
- Linux `libLLVM` discovery: tighter glob, catches `RuntimeError` from component libraries.
- `set_libllvm_path()` invalidates cached backend detection.
- Error line numbers are now correct when `symbols=` or `address=` is used.
- `_split_semicolons`: tracks which quote character opened a string, and backslash escapes (`.ascii "it's; ok"` no longer splits).
- `Disassembler`: `arm64-*` triples skip undecodable bytes in 4-byte steps (was 2); RISC-V uses 2 (was 1).
- `Disassembler`: argtypes and disassembler init are redone after `set_libllvm_path()` loads a new library.
- `Assembler.default_verify` sets the default for `verify=`.
- Apple and Windows triples are assembled as ELF internally. Mach-O left every branch to a local label unrelocated (`jmp end` -> `e9 00000000`; `cbz`/`adr` errored), and COFF crashed on branches to `.set` symbols.
- `asm_each()` assembles once with per-line marker labels instead of cumulatively. Forward references and branch relaxation now match `asm()` (previously `jmp end; nop; end: ret` gave a 5-byte `jmp` and dropped `ret`), ARM literal pools are no longer attached to the `ldr`, and it is O(1) assemblies instead of O(n).
- `verify=True` works with x86 relative branches (checked by decoding) and AT&T syntax, and reports undecodable output as such.
- `Assembler(backend=...)` rejects unknown names with `ValueError`.
- `Disassembler` contexts are freed on garbage collection; `Assembler.close()` also closes its verifier.
- CI: `pytest` moved to the `dev` dependency group, so `uv sync --group dev` installs it.

### Refactoring
- Split `common.py` into `errors.py`, `objfile.py`, `common.py`.
- Removed unused `_active_backends` counter.
- Removed dead imports, commented-out logging.

### Testing
- 5166 tests (up from ~1400). Includes concurrency stress, adversarial symbol injection, round-trip verification, multi-architecture coverage.

## 0.2.0

- LLVM C API backend via ctypes — default when `libLLVM` is available
- `LLVMContextSetDiagnosticHandler` intercepts assembly errors safely
- Thread-safe C API calls (lock-serialized)
- Subprocess fallback when `libLLVM` unavailable
- Architecture profiles: `armv6m`, `armv7m`, `armv8m`, `x86_32` (nyxstone parity)
- `backend=` parameter to force `"capi"` or `"subprocess"`
- Plain class replaces dataclass (mypy-compatible, `triple` is required)
- `__call__` shorthand: `asm("nop")` == `asm.asm("nop")`
- Module-level `flintmc.asm()` with configurable default
- Smart semicolon splitting (preserves quoted strings and comments)
- Bounded LRU cache (4096 entries, thread-safe)
- Error line-number adjustment for injected preamble
- Subprocess timeout (10s)
- `py.typed` marker

## 0.1.0

- Initial release
- Subprocess backend (`llvm-mc`)
- ARM Thumb-2 + x86/x86_64 + AArch64 support
- Cortex-M profiles: `cortex_m7_sp`, `cortex_m7_dp`, `cortex_m4`, `cortex_m33`, `cortex_m0`
- ELF `.text` extraction (32 + 64-bit)
- Auto-detected preamble per architecture
- Keystone parity tests (82 instructions)
- GAS ground-truth tests (118 instructions)
- 1300+ x86 encoding tests
