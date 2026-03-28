# flintmc — Technical Reference

This document provides a deep dive into the architecture, implementation details, and technical design decisions of `flintmc`.

## Architecture

```
flintmc/
├── __init__.py        # Public API: Assembler, AsmError, register_arch_mapping, etc.
├── assembler.py       # Assembler class, ELF parser, preamble detection, caching
├── llvm_capi.py       # LLVM C API backend (ctypes bindings, discovery, TLS)
└── py.typed           # PEP 561 marker
```

The library is designed to be a thin, high-performance bridge between Python and LLVM's Machine Code (MC) layer.

---

## Backend Selection

`flintmc` supports two assembly backends, with automatic detection and fallback:

1.  **C API Backend (`capi`)**: 
    - **Method**: Direct in-process calls to `libLLVM` via `ctypes`.
    - **Performance**: Extremely fast (~0.15ms per call).
    - **Concurrency**: True parallel assembly using **Thread-Local Storage (TLS)**. Each thread manages its own private LLVM context and TargetMachine.
2.  **Subprocess Backend (`subprocess`)**:
    - **Method**: Pipes assembly through the `llvm-mc` binary.
    - **Role**: Reliable fallback when the shared library is missing or for environments where subprocesses are preferred.

---

## C API Implementation

### The Assembly Pipeline
LLVM's C API does not expose a direct "assemble string" function. `flintmc` implements a high-performance workaround using module-level inline assembly:

1.  **Context**: A private `LLVMContext` is created per thread.
2.  **Module**: A throwaway `LLVMModule` is created in that context.
3.  **Injection**: `LLVMSetModuleInlineAsm2()` injects the raw assembly source into the module.
4.  **Emission**: `LLVMTargetMachineEmitToMemoryBuffer()` emits the module as an ELF object file in memory.
5.  **Extraction**: A pure-Python ELF parser extracts the `.text` section bytes.

### C API Functions Used

| Function | Purpose |
| :--- | :--- |
| `LLVMContextSetDiagnosticHandler` | Intercepts `exit()` calls and captures errors. |
| `LLVMSetModuleInlineAsm2` | Inject length-parameterized assembly source. |
| `LLVMTargetMachineEmitToMemoryBuffer` | The core "string-to-ELF" engine. |
| `LLVMCreateTargetMachine` | Creates the architecture-specific generator. |

---

## Robust LLVM Discovery

`flintmc` implements a sophisticated discovery strategy to find LLVM across diverse environments:

### Discovery Priority:
1.  **`LLVM_PATH`**: If set, `flintmc` checks the expected subdirectories (e.g., `/lib` for Unix, `/bin` for Windows).
2.  **Dynamic PATH Discovery**:
    - **Windows**: Finds `llvm-mc.exe` via `PATH` and automatically checks the same directory for `LLVM-C.dll` or `libLLVM.dll`.
    - **Linux**: Uses `glob` to search for versioned libraries (e.g., `libLLVM-18.so`) in standard and multiarch directories (including `/usr/lib64`).
    - **macOS**: Checks Homebrew (Apple Silicon and Intel) and system paths.
3.  **Programmatic Set**: `flintmc.set_libllvm_path(path)` allows explicit runtime configuration.

---

## Technical Error Reporting

`flintmc` prioritizes technical transparency for power users. Every failure point identifies the specific LLVM component involved:

-   **Function Verification**: Every required C API function is verified during library load. Missing functions raise: `"Required LLVM function 'LLVMContextCreate' not found"`.
-   **Target Initialization**: Missing architecture symbols raise: `"LLVM symbol 'LLVMInitializeARMAsmParser' not found. Is ARM support enabled in your LLVM build?"`.
-   **Allocation Failures**: Object creation failures report the specific function: `"LLVMModuleCreateWithNameInContext() failed"`.
-   **Assembly Errors**: Captured LLVM diagnostics are parsed and returned with adjusted line numbers (accounting for injected preambles).

---

## Thread Safety and TLS

Unlike many LLVM wrappers that use a global lock, `flintmc` uses **Thread-Local Storage (TLS)** for the C API backend. 

-   Each thread receives its own `LlvmCApiBackend` instance.
-   Each instance owns a private `LLVMContext`, `LLVMTargetMachine`, and diagnostic callback.
-   This enables **true concurrent assembly** across multiple CPU cores without contention or memory corruption.

---

## Architecture and Triple Mapping

The `_TRIPLE_TO_ARCH` mapping translates common triple prefixes to internal LLVM architecture names.

-   **Supported out-of-the-box**: x86, x86_64, ARM, Thumb, AArch64, RISC-V, WebAssembly, PowerPC, Sparc, SystemZ (s390x), Mips, and LoongArch.
-   **Extensibility**: Users can register custom mappings via **`register_arch_mapping(prefix, llvm_name)`**.

---

## ctypes and Memory Safety

-   **Pointer Truncation**: `flintmc` explicitly declares all `.argtypes` and `.restype` for every LLVM function. This prevents the common "pointer truncation" segfault on 64-bit platforms (where `ctypes` otherwise defaults to 32-bit `c_int`).
-   **Resource Finalization**: All LLVM objects (`Context`, `Module`, `TargetMachine`, `MemoryBuffer`) are proactively disposed of in both success and error paths.
-   **Interpreter Teardown**: The library safely handles Python's non-deterministic module cleanup to prevent `NameError` warnings during process exit.
