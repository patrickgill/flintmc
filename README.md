# Flint MC

LLVM-based multi-architecture assembler for Python. Drop-in replacement for keystone-engine.

Calls directly into the LLVM already installed on your system. When you update LLVM, flintmc picks up new instruction support automatically.

Supports any target your LLVM supports: x86, x86_64, ARM Thumb-2, AArch64, RISC-V, etc. Pure Python via ctypes, no vendored code, no compiled extensions.

## Install

```
brew install llvm     # macOS
apt install llvm      # Linux

pip install flintmc   # or: uv add flintmc
```

## Usage

```python
from flintmc import Assembler

# Pick a target profile
asm = Assembler.x86_64()
asm("mov rax, rbx; ret")          # b'\x48\x89\xd8\xc3'
asm("syscall")                    # b'\x0f\x05'

asm = Assembler.cortex_m7_dp()
asm("mrs r0, PRIMASK")            # b'\xef\xf3\x10\x80'
asm("vfma.f32 s0, s1, s2")       # b'\xa0\xee\x81\x0a'

asm = Assembler.aarch64()
asm("mov x0, #42; ret")
```

Multi-instruction with labels, literal pools, directives — all passed through to LLVM:

```python
asm = Assembler.cortex_m7_dp()
asm("""
    loop:
        subs r0, #1
        bne loop
""")

asm("ldr r0, =0xDEADBEEF\n.ltorg")
asm("ite eq\nmoveq r0, #1\nmovne r0, #0")
```

Custom targets:

```python
asm = Assembler(triple="riscv64", cpu="generic-rv64", features="+m,+a,+f,+d")
asm("addi x1, x0, 42")

# AT&T syntax for x86
att = Assembler(triple="x86_64", preamble="")
att("movq %rbx, %rax")
```

## Profiles

```python
# x86
Assembler.x86_64()             # x86-64, Intel syntax
Assembler.x86_32()             # x86 32-bit, Intel syntax (alias: i686)

# ARM — architecture level
Assembler.armv6m()             # ARMv6-M (Cortex-M0/M0+), Thumb-1
Assembler.armv7m()             # ARMv7-M (Cortex-M3/M4/M7), Thumb-2
Assembler.armv8m()             # ARMv8-M Mainline (Cortex-M33/M55), TrustZone

# ARM — specific CPU + FPU
Assembler.cortex_m7_sp()       # Cortex-M7, FPv5 single-precision — rejects .f64
Assembler.cortex_m7_dp()       # Cortex-M7, FPv5 single + double precision
Assembler.cortex_m4()          # Cortex-M4, FPv4 single-precision
Assembler.cortex_m33()         # Cortex-M33, FPv5-SP + DSP + TrustZone
Assembler.cortex_m0()          # Cortex-M0, Thumb-1 only, no FPU

# AArch64
Assembler.aarch64()            # ARMv8-A 64-bit
```

## Error handling

```python
from flintmc import AsmError

try:
    asm("bad_instruction")
except AsmError as e:
    print(e)  # LLVM diagnostic with adjusted line numbers
```

## How it works

**C API backend (default):** Assembly source is injected into a throwaway LLVM module via `LLVMSetModuleInlineAsm2()`, emitted as an ELF object via `LLVMTargetMachineEmitToMemoryBuffer()`, and the `.text` section is extracted in pure Python. All via ctypes into your system's `libLLVM`. A `LLVMContextSetDiagnosticHandler` intercepts assembly errors so they become `AsmError` exceptions.

**Subprocess backend (fallback):** If `libLLVM` isn't found, falls back to piping through `llvm-mc -filetype=obj -o -`.

Force a backend with `backend="capi"` or `backend="subprocess"`.

Results are cached (thread-safe LRU) — repeated assembly of the same source string is free.

## LLVM compatibility

Tested on **LLVM 22.1.1** (Homebrew, macOS ARM64).

**C API backend** relies on these functions:

| Function | Since |
|----------|-------|
| `LLVMContextSetDiagnosticHandler` | LLVM 3.5 |
| `LLVMCreateTargetDataLayout` | LLVM 3.9 |
| `LLVMSetModuleDataLayout` | LLVM 3.9 |
| `LLVMSetModuleInlineAsm2` | LLVM 7.0 |

Minimum: **LLVM 7.0+** (bottleneck is `LLVMSetModuleInlineAsm2`).

**Subprocess backend** uses `llvm-mc`, included in LLVM since 3.0 (2011).

## Platform

Tested on macOS (Apple Silicon + Intel) and Linux. Windows support is planned — `llvm-mc` needs to be on PATH.

## Testing

```
uv sync --group dev
uv run pytest                                       # core + x86 tests (1400+)

uv sync --group compat
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
uv run pytest tests/test_thumb2_vs_keystone.py      # keystone parity

uv run pytest tests/test_thumb2_vs_gas.py           # GAS ground-truth (needs arm-none-eabi-as)
```

## Requirements

- Python >= 3.10
- LLVM 7.0+ (`libLLVM` shared library + `llvm-mc` binary)
