# Assembly Comparison: flintmc, Nyxstone, and Keystone

This document outlines the technical differences between `flintmc` and established assembly frameworks like **Nyxstone** and **Keystone**, specifically focusing on features that those projects provide which are currently outside the scope of `flintmc`.

## Overview

| Feature | flintmc | Nyxstone | Keystone |
| :--- | :--- | :--- | :--- |
| **Engine** | LLVM C API / `llvm-mc` | LLVM C++ Internals | LLVM (Internal Fork) |
| **Primary Goal** | Lightweight, Fast Python Assembler | Advanced Binary Rewriting | Universal Assembly Engine |
| **Implementation** | Pure Python + `ctypes` | C++ Extension | C library + Bindings |
| **Disassembly** | No | Yes | No (uses Capstone) |

---

## 1. What Nyxstone has over flintmc

**Nyxstone** is a specialized library developed by Emproof for security researchers and binary engineers. Because it is a C++ extension that hooks directly into LLVM's internal C++ classes (rather than the stable C API used by `flintmc`), it offers several "power user" features:

### Advanced Symbol and Label Mapping
Nyxstone's standout feature is the ability to resolve external symbols during assembly.
- **External Maps**: You can pass a Python dictionary of `{ "my_func": 0x12345678 }` to Nyxstone. It will correctly resolve `call my_func` to that specific address.
- **Base Addressing**: You can specify the `PC` (Program Counter) where the code will eventually reside. This ensures that position-relative instructions (like `adr` on AArch64 or `rip`-relative on x86_64) are calculated with bit-perfect accuracy for that specific memory location.
- **In `flintmc`**: We rely on LLVM's internal label resolution. While internal labels (e.g., `loop: ... jmp loop`) work perfectly, there is no direct way via the C API to inject external absolute addresses into the middle of a string-based assembly block.

### Integrated Disassembly
Nyxstone is a "bi-directional" engine. It can disassemble raw bytes into assembly strings and provide detailed metadata about every instruction (registers accessed, memory operands, etc.). `flintmc` is strictly an assembler.

### Low-Level MC Layer Access
By using the C++ API, Nyxstone can access internal LLVM optimizations and "hidden" flags that are not exposed in the stable C API. This allows for extremely fine-grained control over the generation of instruction encodings that might be required for exotic obfuscation or patching.

---

## 2. What Keystone has over flintmc

**Keystone** is the industry standard for lightweight assembly, though it is based on a significantly older version of LLVM (v3.x/v4.x).

### Wide Architecture Support
Because Keystone is an independent C library, it has been ported to almost every imaginable platform and OS. While `flintmc` supports any architecture LLVM supports (x86, ARM, AArch64, RISC-V, MIPS, etc.), it requires a `libLLVM` or `llvm-mc` installation on the host system. Keystone carries its engine with it.

### Statement-Level Control
Keystone allows you to assemble instructions one-by-one and maintain state between them more easily than the "module-based" approach used by the LLVM C API.

### Significant Ecosystem
As the older, more established project, Keystone has thousands of community plugins, integrations with debuggers (OllyDbg, x64dbg), and extensive documentation for edge-case assembly requirements.

---

## 3. Why choose flintmc instead?

Despite lacking the advanced binary-patching features of Nyxstone, `flintmc` has several advantages for standard development:

1.  **Zero-Compilation Install**: Unlike Nyxstone or Keystone's Python bindings, `pip install flintmc` is a pure-Python wheel. It does not require a local C++ compiler (no `gcc` or `clang` invocation) during installation. It simply "plugs in" to the pre-compiled `libLLVM` already provided by your OS package manager (`brew`, `apt`, etc.).
2.  **Modern LLVM**: `flintmc` uses whatever LLVM is on your system (LLVM 15, 16, 17, 18+). Keystone is stuck on a decade-old fork of LLVM, meaning it lacks support for modern instructions (AVX-512, newer ARM extensions, etc.).
3.  **Performance**: `flintmc` is tuned for speed. By reusing LLVM modules and using Thread-Local Storage, it can achieve ~0.15ms per assembly call, making it suitable for high-throughput tasks like JIT compilation or massive test-case generation.
4.  **Simplicity**: The codebase is small, readable, and written in pure Python. It is significantly easier to audit and customize for specific project needs.
