# flintmc

Multi-architecture assembler backed by [llvm-mc](https://llvm.org/docs/CommandGuide/llvm-mc.html).

Works for any target LLVM supports — x86, x86_64, ARM Thumb-2, AArch64, RISC-V, etc. Built as a modern replacement for [keystone-engine](https://www.keystone-engine.org/) that stays current with LLVM instead of lagging behind by a decade.

Single subprocess call per assembly, ELF `.text` extraction in pure Python, no temp files, no native dependencies beyond LLVM itself.

## Install

```
brew install llvm     # macOS
apt install llvm      # Linux

pip install flintmc   # or: uv add flintmc
```

## Usage

```python
from flintmc import Assembler

# x86_64
x64 = Assembler.x86_64()
x64.asm("mov rax, rbx; ret")     # b'\x48\x89\xd8\xc3'
x64.asm("syscall")               # b'\x0f\x05'

# x86 32-bit
x86 = Assembler.i686()
x86.asm("int 0x80")              # b'\xcd\x80'

# ARM Cortex-M7 (default)
asm = Assembler()
asm.asm("mrs r0, PRIMASK")       # b'\xef\xf3\x10\x80'
asm.asm("vfma.f32 s0, s1, s2")   # b'\xa0\xee\x81\x0a'

# AArch64
a64 = Assembler.aarch64()
a64.asm("ret")
```

Multi-instruction assembly (semicolons or newlines), labels, literal pools, all standard assembler directives — passed straight through to LLVM's MC layer:

```python
asm = Assembler()

asm.asm("""
    loop:
        subs r0, #1
        bne loop
""")

asm.asm("ldr r0, =0xDEADBEEF\n.ltorg")  # literal pool expansion

asm.asm("ite eq\nmoveq r0, #1\nmovne r0, #0")  # IT blocks

asm.asm("b .", addr=0x100)  # PC-relative at specific address
```

Results are cached — identical `(source, addr)` pairs return the same bytes without re-invoking llvm-mc.

## Profiles

Preset configurations for common targets.

### x86

```python
x64 = Assembler.x86_64()   # Intel syntax by default
x86 = Assembler.i686()     # 32-bit, Intel syntax

# AT&T syntax
att = Assembler(triple="x86_64", cpu="", features="", preamble="")
att.asm("movq %rbx, %rax")
```

### ARM Cortex-M

```python
Assembler.cortex_m7_sp()   # FPv5 single-precision (i.MX RT1062, Teensy 4.x)
Assembler.cortex_m7_dp()   # FPv5 single + double precision
Assembler.cortex_m4()      # FPv4 single-precision
Assembler.cortex_m33()     # ARMv8-M, TrustZone, DSP
Assembler.cortex_m0()      # Thumb-1 only, no FPU
```

SP-only profiles reject double-precision instructions at assembly time.

### AArch64

```python
Assembler.aarch64()
```

## Custom targets

For anything not covered by presets, pass LLVM triple/cpu/features directly:

```python
asm = Assembler(
    triple="riscv64",
    cpu="generic-rv64",
    features="+m,+a,+f,+d",
)
```

### Preamble

Each architecture auto-detects sensible preamble directives — ARM gets `.syntax unified` / `.thumb`, x86 gets `.intel_syntax noprefix`. Override with `preamble=`:

```python
# Custom preamble
raw = Assembler(triple="thumbv7em-none-eabi", cpu="cortex-m7",
                features="+fp-armv8,+fp64", preamble="")
```

## Errors

```python
from flintmc import AsmError

try:
    asm.asm("bad_instruction")
except AsmError as e:
    print(e)  # LLVM diagnostic with <stdin> prefix stripped
```

## How it works

1. Assembly source is wrapped with auto-detected preamble directives
2. Piped to `llvm-mc -triple=... -filetype=obj -o -` via subprocess
3. `.text` section extracted from the ELF object (32 or 64-bit) in pure Python
4. Raw machine code bytes returned

## Platform

Tested on macOS (Apple Silicon + Intel) and Linux. Windows support is planned — `llvm-mc` needs to be on PATH.

## Testing

```
uv sync --group dev
uv run pytest                              # core + x86 tests

uv sync --group compat
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
uv run pytest tests/test_vs_keystone.py    # keystone parity

uv run pytest tests/test_vs_gas.py         # GAS ground-truth (if arm-none-eabi-as installed)
```

## Requirements

- Python >= 3.13
- LLVM (`llvm-mc` binary) — `brew install llvm` / `apt install llvm`
