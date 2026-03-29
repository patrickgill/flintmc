# Assembly Comparison: flintmc, Nyxstone, and Keystone

## Overview

| Feature | flintmc | Nyxstone | Keystone |
| :--- | :--- | :--- | :--- |
| **Engine** | LLVM C API / `llvm-mc` | LLVM C++ Internals | LLVM (Internal Fork) |
| **Install** | `pip install` — pure Python | C++ extension, needs compiler | C library + bindings |
| **LLVM** | System LLVM (7.0+) | Requires LLVM 15-18 | Frozen fork (~3.x) |
| **Disassembly** | Yes | Yes | No (uses Capstone) |
| **Symbol injection** | Yes (`.set` + relocation patching) | Yes (native MCSymbol) | No |
| **Base address** | Yes (`.org`, max 1 MiB) | Yes (native, unlimited) | No |
| **Per-instruction info** | Yes (`asm_each()`) | Yes | Count only |
| **Round-trip verify** | Yes (`verify=True`) | No | No |
| **Concurrency** | True parallel (TLS) | Global lock | Global lock |
| **Object formats** | ELF + Mach-O + COFF | N/A (no object files) | N/A (raw output) |
| **Context manager** | Yes | No | No |

---

## How each tool works

**flintmc**: Injects assembly into an LLVM module via the C API, emits an object file in memory, and extracts the `.text` section in pure Python. For symbol injection, applies ELF relocations to resolve branch targets. Falls back to `llvm-mc` subprocess when `libLLVM` is unavailable.

**Nyxstone**: Hooks directly into LLVM's internal C++ `MCAssembler` / `MCStreamer` classes. Assembles each instruction and reads encoded bytes from `MCInst`. Never produces an object file. This gives it native symbol resolution and unlimited base addresses, but requires a C++ build step.

**Keystone**: Uses a forked, vendored copy of LLVM (~v3.x). Ships as a standalone C library with language bindings. No system LLVM dependency, but no modern instruction support.

---

## What Nyxstone has over flintmc

### Native symbol resolution
Nyxstone resolves symbols by injecting them into LLVM's `MCSymbol` table. This works uniformly across all architectures and instruction types. flintmc uses `.set` directives + relocation patching, which works for all practical cases on ELF targets but has a known LLVM crash on Windows COFF targets for branch instructions.

### Unlimited base address
Nyxstone sets the assembly origin by configuring `MCAssembler` directly. flintmc uses `.org` which pads the object file — capped at 1 MiB to avoid excessive memory use.

### Direct MC layer access
Nyxstone can access internal LLVM optimizations and flags not exposed in the C API. This allows exotic encoding control that flintmc cannot replicate.

---

## What Keystone has over flintmc

### Self-contained
Keystone ships its own LLVM. No system dependency. flintmc requires `libLLVM` or `llvm-mc` on the host.

### Ecosystem
Thousands of community plugins, debugger integrations (OllyDbg, x64dbg), and extensive edge-case documentation.

---

## What flintmc has over both

1.  **Zero-compilation install**: `pip install flintmc` — no C++ compiler needed.
2.  **Modern LLVM**: Uses whatever LLVM is on your system (7.0 through 22+). Automatically picks up new instruction support.
3.  **True concurrency**: Thread-Local Storage instead of global locks.
4.  **Round-trip verification**: `verify=True` catches encoding mismatches that neither Nyxstone nor Keystone detect.
5.  **Disassembly + assembly in one package**: No need for a separate Capstone install.
6.  **Subprocess fallback**: Works even without `libLLVM` (slower, but functional).
7.  **Three object format parsers**: ELF, Mach-O, COFF — covers Linux, macOS, and Windows targets.

---

## Known limitations

- **COFF branch relocations**: LLVM segfaults when emitting `jmp`/`call` to `.set` symbols with COFF output. This is an upstream LLVM bug. Data references (mov, lea) work fine on COFF.
- **Base address limit**: `.org` approach caps at 1 MiB. Use `.org` directly in the source for larger addresses, or compute section-relative offsets in the `symbols` dict.
- **WebAssembly**: LLVM emits wasm binary format, not ELF/Mach-O/COFF. Not supported.
