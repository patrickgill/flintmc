# flintmc

Multi-architecture assembler backed by [llvm-mc](https://llvm.org/docs/CommandGuide/llvm-mc.html).

Works for any target LLVM supports — ARM Thumb-2, x86, x86_64, AArch64, RISC-V, etc. Built to replace [keystone-engine](https://www.keystone-engine.org/) with something that doesn't lag behind LLVM by a decade.

Keystone's ARM assembler is based on an old LLVM fork that can't handle M-class special registers (`PRIMASK`, `BASEPRI`, `FAULTMASK`) in MSR/MRS, is missing FPv5 fused multiply-accumulate (`VFMA`, `VFMS`), and chokes on `UDIV`/`SDIV` in Thumb mode. flintmc delegates to your system's current `llvm-mc` — whatever LLVM supports, flintmc supports.

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

# ARM Cortex-M7 (default)
asm = Assembler()
asm.asm("mrs r0, PRIMASK")       # b'\xef\xf3\x10\x80'
asm.asm("vfma.f32 s0, s1, s2")   # b'\xa0\xee\x81\x0a'
asm.asm("udiv r0, r1, r2")       # b'\xb1\xfb\xf2\xf0'

# x86_64
x64 = Assembler.x86_64()
x64.asm("mov rax, rbx; ret")     # b'\x48\x89\xd8\xc3'
x64.asm("syscall")               # b'\x0f\x05'

# x86 32-bit
x86 = Assembler.i686()
x86.asm("int 0x80")              # b'\xcd\x80'

# AArch64
a64 = Assembler.aarch64()
a64.asm("ret")
```

Multi-instruction assembly (semicolons or newlines), labels, literal pools, all standard assembler directives — passed straight through to LLVM's MC layer:

```python
asm.asm("""
    loop:
        subs r0, #1
        bne loop
""")

asm.asm("ldr r0, =0xDEADBEEF\n.ltorg")  # literal pool expansion

asm.asm("nop.n")   # 16-bit narrow
asm.asm("nop.w")   # 32-bit wide

asm.asm("ite eq\nmoveq r0, #1\nmovne r0, #0")  # IT blocks

asm.asm("b .", addr=0x1000)  # PC-relative at specific address
```

Results are cached — identical `(source, addr)` pairs return the same bytes without re-invoking llvm-mc.

## Profiles

Preset configurations for common targets. Each documents the equivalent `arm-none-eabi-as` or GAS flags.

### ARM Cortex-M

```python
# Cortex-M7 single-precision (i.MX RT1062, Teensy 4.x)
# = arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-sp-d16
asm = Assembler.cortex_m7_sp()
asm.asm("vadd.f32 s0, s1, s2")   # works
asm.asm("vadd.f64 d0, d1, d2")   # AsmError — rejected

# Cortex-M7 double-precision
# = arm-none-eabi-as -mcpu=cortex-m7 -mfpu=fpv5-d16
asm = Assembler.cortex_m7_dp()

# Cortex-M4 single-precision
# = arm-none-eabi-as -mcpu=cortex-m4 -mfpu=fpv4-sp-d16
asm = Assembler.cortex_m4()

# Cortex-M33 (ARMv8-M, TrustZone, DSP)
# = arm-none-eabi-as -mcpu=cortex-m33 -mfpu=fpv5-sp-d16
asm = Assembler.cortex_m33()
asm.asm("sg")           # Secure Gateway
asm.asm("tt r0, r1")    # Test Target

# Cortex-M0/M0+ (Thumb-1 only, no FPU)
asm = Assembler.cortex_m0()
```

### x86

```python
asm = Assembler.x86_64()    # Intel syntax by default
asm = Assembler.i686()      # 32-bit, Intel syntax
```

### AArch64

```python
asm = Assembler.aarch64()
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
# AT&T syntax for x86
att = Assembler(triple="x86_64", cpu="", features="", preamble="")
att.asm("movq %rbx, %rax")

# Disable all preamble
raw = Assembler(triple="thumbv7em-none-eabi", cpu="cortex-m7",
                features="+fp-armv8,+fp64", preamble="")
```

### FPU feature mapping (ARM)

| GAS `-mfpu=`       | LLVM features                          | What it means                    |
| ------------------- | -------------------------------------- | -------------------------------- |
| `fpv5-d16`          | `+fp-armv8,+fp64`                      | FPv5, single + double precision  |
| `fpv5-sp-d16`       | `+fp-armv8d16sp,-fp64,-fpregs64`       | FPv5, single precision only      |
| `fpv4-sp-d16`       | `+vfp4,-fp64,-fpregs64`               | FPv4, single precision only      |
| *(none)*            | `""` (empty)                           | No FPU                           |

## Errors

```python
from flintmc import AsmError

try:
    asm.asm("vfma.f32 s0, s1, s99")
except AsmError as e:
    print(e)  # LLVM diagnostic with <stdin> prefix stripped
```

## How it works

1. Assembly source is wrapped with auto-detected preamble directives
2. Piped to `llvm-mc -triple=... -filetype=obj -o -` via subprocess
3. `.text` section extracted from the ELF object (32 or 64-bit) in pure Python
4. Raw machine code bytes returned

## Testing

```
uv sync --group dev
uv run pytest                              # core + x86 tests

uv sync --group compat
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
uv run pytest tests/test_vs_keystone.py    # keystone parity

uv run pytest tests/test_vs_gas.py         # GAS ground-truth (if arm-none-eabi-as installed)
```

The GAS comparison suite validates byte-for-byte parity with `arm-none-eabi-as` across 118 ARM instructions. If flintmc and GAS agree, the encoding is correct.

## Requirements

- Python >= 3.13
- LLVM (`llvm-mc` binary) — `brew install llvm` / `apt install llvm`
