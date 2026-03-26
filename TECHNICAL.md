# flintmc — Technical Reference

## Architecture

```
flintmc/
├── __init__.py        # Public API: Assembler, AsmError, find_llvm_mc, asm()
├── assembler.py       # Assembler class, ELF parser, preamble detection, caching
├── llvm_capi.py       # LLVM C API backend (ctypes bindings)
└── py.typed           # PEP 561 marker
```

875 lines of library code. ~4600 lines of tests.

## Backend Selection

flintmc has two assembly backends. The default is auto-detected at construction time:

1. **C API backend** (`llvm_capi.py`): Calls `libLLVM.dylib`/`.so` via ctypes. In-process, no subprocess spawn. The `Assembler.__init__` calls `try_create_backend()`, which attempts to load `libLLVM`, initialize the target architecture, and create a `TargetMachine`. If any step fails, returns `None` and the assembler falls back to subprocess.

2. **Subprocess backend**: Pipes assembly through `llvm-mc -filetype=obj -o -` and parses the ELF output. Used when `libLLVM` isn't available, or when forced with `backend="subprocess"`.

The backend can be forced explicitly:

```python
Assembler.x86_64(backend="capi")        # force C API
Assembler.x86_64(backend="subprocess")  # force subprocess
Assembler.x86_64()                      # auto-detect (C API if available)
```

## C API Pipeline

LLVM's C API has no direct "assemble this string" function. The MC assembler pipeline (`MCAsmParser`, `MCStreamer`, etc.) is C++ only with no C bindings.

The workaround: `LLVMSetModuleInlineAsm2()` injects raw assembly text into an LLVM IR module as module-level inline asm. `LLVMTargetMachineEmitToMemoryBuffer()` then emits the module as an ELF object file — going through the full MC assembler layer internally. This is the same code path that `__asm__` blocks in C/C++ take through clang.

Per-call flow:

```
LLVMModuleCreateWithNameInContext()    — throwaway module in private context
LLVMSetTarget()                        — set triple on module
LLVMCreateTargetDataLayout()           — get data layout from TM
LLVMSetModuleDataLayout()              — apply to module
LLVMSetModuleInlineAsm2()             — inject assembly source
LLVMTargetMachineEmitToMemoryBuffer() — emit ELF to memory
  → diagnostic handler fires on errors
extract .text from ELF                 — pure Python struct parsing
LLVMDisposeMemoryBuffer()             — free ELF buffer
LLVMDisposeModule()                    — free module
```

The `TargetMachine` is created once and reused across all calls. Only the module is throwaway.

### C API Functions Used

| Function | LLVM Version | Purpose |
|----------|-------------|---------|
| `LLVMContextCreate` | 2.6 | Private context for diagnostic handler |
| `LLVMContextSetDiagnosticHandler` | 3.5 | Intercept assembly errors |
| `LLVMCreateTargetDataLayout` | 3.9 | Get layout from TargetMachine |
| `LLVMSetModuleDataLayout` | 3.9 | Apply layout to module |
| `LLVMSetModuleInlineAsm2` | **7.0** | Inject assembly (length-parameterized) |
| `LLVMModuleCreateWithNameInContext` | 2.8 | Create module in private context |
| `LLVMGetTargetFromTriple` | 2.9 | Target lookup |
| `LLVMCreateTargetMachine` | 3.1 | Create reusable TM |
| `LLVMTargetMachineEmitToMemoryBuffer` | 3.3 | Emit ELF to memory |
| `LLVMGetBufferStart` / `LLVMGetBufferSize` | 2.6 | Read ELF bytes |
| `LLVMGetDiagInfoDescription` / `LLVMGetDiagInfoSeverity` | 3.5 | Read diagnostic details |

Minimum LLVM version: **7.0** (bottleneck: `LLVMSetModuleInlineAsm2`).

## Error Handling

### The exit() problem

LLVM's inline asm path normally calls `exit()` when it encounters an assembly error (invalid instruction, missing feature, etc.). This happens deep in the `AsmPrinter` codegen path and would kill the Python process.

### The solution: LLVMContextSetDiagnosticHandler

Each `LlvmCApiBackend` creates a private `LLVMContext` (not the global one) and installs a diagnostic handler callback via `LLVMContextSetDiagnosticHandler`. When LLVM encounters an error, it calls our handler *instead of* the default handler (which is what calls `exit()`). The handler captures the error message into a list.

After `LLVMTargetMachineEmitToMemoryBuffer()` returns, we check the captured errors list. If non-empty, we raise `AsmError` with the diagnostic message. LLVM may return `rc=0` even when errors were captured (it "succeeded" at emitting garbage), so the errors list is the source of truth, not the return code.

The diagnostic callback must be stored as an instance attribute (`self._diag_callback`) to prevent garbage collection — ctypes doesn't prevent GC of callback objects.

### Error line numbers

Both backends produce errors with `<stdin>:N:` or `<inline asm>:N:` line prefixes. The `_fix_error()` function adjusts line numbers by subtracting the preamble line count, so errors reference the user's source lines, not the internal preamble-prepended source.

## ctypes Safety

### The SIGSEGV problem

On ARM64 macOS (and any LP64 platform), ctypes defaults to `c_int` (32-bit) for all function arguments when `.argtypes` is not declared. This silently truncates 64-bit pointers to 32 bits. LLVM functions receive garbage pointers and segfault.

Crash signature: `EXC_BAD_ACCESS at 0xfffffffff10c4600` — the `0xffffffff` prefix is the sign-extended upper 32 bits of a truncated pointer.

### The fix

All function `.restype` and `.argtypes` are declared in `_declare_argtypes()` immediately after loading `libLLVM`, before any function is called. This is enforced by the `_load_llvm()` flow — the library isn't returned until argtypes are set.

Per-arch init functions (`LLVMInitialize{Arch}TargetInfo`, etc.) get their argtypes set in `_init_target()` since they're discovered dynamically.

## ELF .text Extraction

Both backends produce an ELF object file (32-bit for ARM/RISC-V, 64-bit for x86_64/AArch64). The `.text` section is extracted in pure Python using `struct.unpack_from`:

1. Read `ei_class` (byte 4) to determine 32 vs 64-bit
2. Parse ELF header for section header table offset, entry size, count, string table index
3. Walk section headers, look up names in string table
4. Return the `.text` section's raw bytes

This parser handles only what `llvm-mc` / the C API emits — minimal ELF objects with a single `.text` section. It doesn't handle relocations, multiple text sections, or big-endian ELF.

## Preamble

Each target architecture needs specific assembler directives prepended to the source:

| Triple prefix | Preamble |
|--------------|----------|
| `thumb*` | `.syntax unified\n.thumb` |
| `arm*` | `.syntax unified\n.arm` |
| `x86_64`, `x86-64` | `.intel_syntax noprefix` |
| `i686`, `i386` | `.intel_syntax noprefix\n.code32` |
| everything else | *(none)* |

Auto-detected from the triple in `_default_preamble()`. Override with `preamble=""` to disable, or pass any custom string.

## Semicolon Handling

`source.replace(";", "\n")` would break `.ascii "hello;world"` and comments containing semicolons. The `_split_semicolons()` function handles this:

- Tracks quote state (`"..."`) — semicolons inside quotes are preserved
- Detects comment markers (`@`, `#`, `//`) — semicolons after comments are preserved
- Only bare semicolons between instructions are converted to newlines

## Caching

`Assembler.asm()` maintains a thread-safe LRU cache (`OrderedDict` + `threading.Lock`):

- Cache key: the raw source string
- Cache value: assembled `bytes`
- Max size: 4096 entries
- Eviction: LRU (oldest-accessed entry removed when full)
- Thread safety: lock acquired for cache lookup and insertion; the actual assembly (subprocess or C API call) runs outside the lock

Cache hits return the same `bytes` object (identity, not just equality).

## Thread Safety

- **Cache**: Protected by `threading.Lock` in `Assembler`. Lookup and insertion are serialized; assembly runs unlocked.
- **C API backend**: All LLVM calls are serialized behind a `threading.Lock` in `LlvmCApiBackend`. LLVM contexts are not thread-safe — concurrent module creation/emit will segfault without serialization.
- **Subprocess backend**: Inherently safe — each call is an independent process.
- **Diagnostic handler**: Errors are collected into a list that's protected by the same lock that serializes C API calls. The callback runs inside `LLVMTargetMachineEmitToMemoryBuffer()`, which is already holding the lock.

## Thumb Encoding Width

The C API backend and subprocess backend produce identical encodings for the same source. Both use LLVM's inline asm path, which does not run `MCAssembler::relaxInstruction()` (the Thumb narrow-encoding relaxation pass). The standalone `llvm-mc` tool does run this pass, so its output may differ for some Thumb instructions.

In practice this means `mov r0, #42` emits as `mov.w` (4 bytes) rather than `movs` (2 bytes). Use the explicit narrow mnemonic (`movs`) when size matters. This matches keystone's behavior — both flintmc and keystone produce identical encoding widths for the same mnemonics.

## libLLVM Discovery

`_load_llvm()` in `llvm_capi.py` checks platform-specific paths:

- **macOS**: `/opt/homebrew/opt/llvm/lib/libLLVM.dylib` (Apple Silicon), `/usr/local/opt/llvm/lib/libLLVM.dylib` (Intel)
- **Linux**: `/usr/lib/libLLVM.so`, versioned paths (`/usr/lib/llvm-{18..15}/lib/libLLVM.so`)
- **Windows**: Not yet supported (would need PATH-based discovery)

`find_llvm_mc()` in `assembler.py` discovers the `llvm-mc` binary similarly, with an additional check for versioned binaries on Linux (`/usr/bin/llvm-mc-18`, etc.) and a `shutil.which()` fallback.

## Target Architecture Initialization

LLVM requires per-architecture initialization before any target can be used. The `_init_target()` function calls five init functions per arch:

```
LLVMInitialize{Arch}TargetInfo()
LLVMInitialize{Arch}Target()
LLVMInitialize{Arch}AsmPrinter()
LLVMInitialize{Arch}AsmParser()
LLVMInitialize{Arch}TargetMC()
```

Initialization is idempotent — tracked in a module-level `_initialized_arches` set. The arch name is mapped from the triple prefix via `_TRIPLE_TO_ARCH` (e.g. `thumb*` → `ARM`, `x86_64` → `X86`).

## Test Architecture

| File | Tests | What it covers |
|------|-------|---------------|
| `test_thumb2.py` | ~76 | ARM Thumb-2 instructions, profiles, caching, errors, thread safety, semicolons |
| `test_x86.py` | ~1318 | x86/x86_64 GPR, ALU, SIMD (SSE through AVX2/FMA3), AES-NI, BMI, exact encodings |
| `test_thumb2_vs_gas.py` | ~118 | Byte-for-byte comparison with `arm-none-eabi-as` (ground truth) |
| `test_thumb2_vs_keystone.py` | ~82 | Parity with keystone-engine on common instructions + gap demonstration |
| `test_vs_nyxstone.py` | ~598 | Cross-validation with nyxstone (LLVM 15-18) |
| `test_aarch64.py` | ~1107 | AArch64 instruction coverage |
| `test_riscv.py` | ~887 | RISC-V instruction coverage |
