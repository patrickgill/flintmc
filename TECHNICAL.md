# flintmc — Technical Reference

This document provides a deep dive into the architecture, implementation details, and technical design decisions of `flintmc`.

## Architecture

```
flintmc/
├── __init__.py          # Public API: Assembler, Disassembler, AsmError, etc.
├── assembler.py         # Assembler class, profiles, caching, asm_each, verify
├── disassembler.py      # Disassembler class, LLVMDisasmInstruction bindings
├── errors.py            # AsmError, UnsupportedArchitectureError
├── objfile.py           # ELF/Mach-O/COFF .text extraction + relocation patching
├── common.py            # Source preprocessing, error fixup, preamble defaults
├── llvm_capi.py         # LLVM C API backend (ctypes bindings, discovery, TLS)
├── llvm_subprocess.py   # llvm-mc subprocess backend + discovery
└── py.typed             # PEP 561 marker
```

The library is designed to be a thin, high-performance bridge between Python and LLVM's Machine Code (MC) layer.

---

## Backend Selection

`flintmc` supports two assembly backends, with automatic detection and fallback:

1.  **C API Backend (`capi`)**:
    - **Method**: Direct in-process calls to `libLLVM` via `ctypes`.
    - **Performance**: ~0.15ms per call.
    - **Concurrency**: True parallel assembly using **Thread-Local Storage (TLS)**. Each thread manages its own private LLVM context and TargetMachine.
2.  **Subprocess Backend (`subprocess`)**:
    - **Method**: Pipes assembly through the `llvm-mc` binary.
    - **Role**: Reliable fallback when the shared library is missing.

The **Disassembler** requires the C API backend. There is no subprocess fallback for disassembly.

---

## C API Assembly Pipeline

LLVM's C API has no direct "assemble this string" function. `flintmc` implements a workaround using module-level inline assembly:

1.  **Context**: A private `LLVMContext` is created per thread.
2.  **Module**: A throwaway `LLVMModule` is created in that context.
3.  **Injection**: `LLVMSetModuleInlineAsm2()` injects the raw assembly source into the module.
4.  **Emission**: `LLVMTargetMachineEmitToMemoryBuffer()` emits the module as an object file in memory.
5.  **Extraction**: A pure-Python parser extracts the `.text` / `__text` section bytes.
6.  **Relocation**: For ELF objects, `.rela.text` / `.rel.text` relocations are applied so that branch instructions to `.set` symbols resolve correctly.

**ASSUMPTION**: The module contains no IR functions or globals — only the inline asm string — so no codegen state accumulates between emissions. If a future LLVM version changes this, fall back to per-call modules.

### C API Functions Used

| Function | Purpose |
| :--- | :--- |
| `LLVMContextSetDiagnosticHandler` | Intercepts `exit()` calls and captures errors. |
| `LLVMSetModuleInlineAsm2` | Inject length-parameterized assembly source. |
| `LLVMTargetMachineEmitToMemoryBuffer` | The core "string-to-object" engine. |
| `LLVMCreateTargetMachine` | Creates the architecture-specific generator. |
| `LLVMCreateDisasmCPUFeatures` | Creates a disassembler context. |
| `LLVMDisasmInstruction` | Decodes one instruction from raw bytes. |

---

## C API Disassembly Pipeline

Uses `LLVMCreateDisasmCPUFeatures` to create a thread-local disassembler context, then `LLVMDisasmInstruction` to decode each instruction sequentially. Intel syntax is enabled for x86 targets via `LLVMSetDisasmOptions`.

---

## Object File Parsing

`flintmc` supports three object file formats, all parsed in pure Python:

| Format | Detection | .text section name |
| :--- | :--- | :--- |
| ELF (32/64-bit) | `\x7fELF` magic | `.text` |
| Mach-O (32/64-bit) | `0xFEEDFACE` / `0xFEEDFACF` magic | `__text` |
| COFF | Machine type field (0x8664, 0x014c, etc.) | `.text` |

Both big-endian and little-endian ELF are supported (detected via `EI_DATA` byte).

### ELF Relocation Patching

When LLVM emits branch instructions (x86 `jmp`/`call`) to `.set` symbols, it generates relocation entries instead of encoding the offset inline. `flintmc` parses these and applies them:

- **ELF64**: `.rela.text` (explicit addend) — `R_X86_64_PC32`, `R_X86_64_PLT32`
- **ELF32**: `.rel.text` (implicit addend) — `R_386_PC32`

Formula: `*(int32*)(code + r_offset) = S + A - P` where S = symbol value, A = addend, P = r_offset.

AArch64, ARM, and RISC-V typically resolve `.set` symbols inline without generating relocations.

---

## Symbol Injection

The `symbols=` parameter injects `.set` directives before the assembly source:

```python
asm("bl handler", symbols={"handler": 0x8000})
# Becomes: .set handler, 0x8000\nbl handler
```

Symbol names are validated against `[A-Za-z_.][A-Za-z0-9_.$]*` to prevent assembly injection.

For **data references** (mov, ldr), the value is used as an absolute address. For **branch instructions** on architectures without relocation patching, the value is a section-relative byte offset. On x86 (where relocations are applied), branch targets resolve to absolute addresses.

---

## Robust LLVM Discovery

### Discovery Priority:
1.  **`LLVM_PATH`**: If set, checks expected subdirectories.
2.  **Dynamic PATH Discovery**:
    - **Windows**: Finds `llvm-mc.exe` via `PATH` and checks for `LLVM-C.dll`.
    - **Linux**: Searches for `libLLVM.so` and versioned `libLLVM-N.so` in standard and multiarch directories. Component libraries (e.g. `libLLVMCore.so`) are skipped.
    - **macOS**: Checks Homebrew (Apple Silicon and Intel) and system paths.
3.  **Programmatic Set**: `flintmc.set_libllvm_path(path)` for explicit runtime configuration. Invalidates cached backend detection.

---

## Thread Safety and TLS

`flintmc` uses **Thread-Local Storage (TLS)** for both backends:

-   Each thread receives its own `LlvmCApiBackend` and `Disassembler` context.
-   Each instance owns a private `LLVMContext`, `LLVMTargetMachine`, and diagnostic callback.
-   This enables **true concurrent assembly** across multiple CPU cores without contention.

The assembly result cache is shared across threads, protected by a lock.

---

## Architecture and Triple Mapping

The `_TRIPLE_TO_ARCH` mapping translates common triple prefixes to internal LLVM architecture names. Sorted by prefix length (longest first) so `arm64` matches before `arm`.

-   **Supported**: x86, x86_64, ARM, Thumb, AArch64, RISC-V, PowerPC, Sparc, SystemZ (s390x), Mips, LoongArch, AVR, BPF, MSP430.
-   **Extensibility**: `register_arch_mapping(prefix, llvm_name)` and `register_default_preamble(prefix, preamble)`.
-   **WebAssembly**: Omitted — LLVM emits wasm binary format, not ELF/Mach-O/COFF.

---

## ctypes and Memory Safety

-   **Pointer Truncation**: All `.argtypes` and `.restype` declared for every LLVM function to prevent 64-bit pointer truncation on LP64 platforms.
-   **Resource Finalization**: `weakref.finalize` ensures LLVM objects are disposed even during non-deterministic Python shutdown.
-   **Context Manager**: `Assembler` and `Disassembler` support `with` statements for deterministic cleanup.
-   **In-Memory Only**: Object files are emitted to `LLVMMemoryBuffer` (heap), never to disk. Subprocess backend uses stdin/stdout pipes.

---

## Error Reporting

-   **Function Verification**: Every required C API function is verified during library load. Missing functions raise: `"Required LLVM function 'LLVMContextCreate' not found"`.
-   **Target Initialization**: Missing architecture symbols raise: `"LLVM symbol 'LLVMInitializeARMAsmParser' not found. Is ARM support enabled in your LLVM build?"`.
-   **Assembly Errors**: Captured LLVM diagnostics with line numbers adjusted for injected preambles and symbols.
-   **Round-Trip Verification**: `verify=True` disassembles output and re-assembles to detect encoding mismatches.
