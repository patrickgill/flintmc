# flintmc

Multi-architecture assembler backed by [llvm-mc](https://llvm.org/docs/CommandGuide/llvm-mc.html).

Works for any target LLVM supports — x86, x86_64, ARM Thumb-2, AArch64, RISC-V, etc. Built as a modern replacement for [keystone-engine](https://www.keystone-engine.org/) that stays current with LLVM instead of lagging behind by a decade.

Single subprocess call per assembly, ELF `.text` extraction in pure Python, no temp files, no native dependencies beyond LLVM itself.

```python
from flintmc import Assembler

asm = Assembler.x86_64()
asm.asm("mov rax, rbx; ret")     # b'\x48\x89\xd8\xc3'
```

### Basic usage

Pick a profile, call `asm()`:

```python
from flintmc import Assembler

# x86_64 — Intel syntax by default
x64 = Assembler.x86_64()
x64.asm("mov rax, rbx; ret")     # b'\x48\x89\xd8\xc3'
x64.asm("syscall")               # b'\x0f\x05'

# x86 32-bit
x86 = Assembler.i686()
x86.asm("int 0x80")              # b'\xcd\x80'

# ARM Cortex-M7 (double precision FPU)
arm = Assembler.cortex_m7_dp()
arm.asm("mrs r0, PRIMASK")       # b'\xef\xf3\x10\x80'
arm.asm("vfma.f32 s0, s1, s2")   # b'\xa0\xee\x81\x0a'

# AArch64
a64 = Assembler.aarch64()
a64.asm("ret")                   # b'\xc0\x03\x5f\xd6'
```

The callable shorthand also works — `asm("...")` and `asm.asm("...")` are equivalent:

```python
asm = Assembler.cortex_m7_dp()
asm("mrs r0, PRIMASK")           # b'\xef\xf3\x10\x80'
```

Or use the module-level convenience API:

```python
import flintmc

flintmc.default = flintmc.Assembler.x86_64()
flintmc.asm("nop")               # b'\x90'
```

### Multi-instruction assembly

Semicolons or newlines separate instructions. Labels, literal pools, and all standard assembler directives are passed straight through to LLVM's MC layer:

```python
asm = Assembler.cortex_m7_dp()

# Labels and branches
asm.asm("""
    loop:
        subs r0, #1
        bne loop
""")

# Literal pool expansion
asm.asm("ldr r0, =0xDEADBEEF\n.ltorg")

# IT blocks
asm.asm("ite eq\nmoveq r0, #1\nmovne r0, #0")
```

### Custom targets

For anything not covered by presets, pass LLVM triple/cpu/features directly:

```python
rv = Assembler(
    triple="riscv64",
    cpu="generic-rv64",
    features="+m,+a,+f,+d",
)
rv.asm("addi x1, x0, 42")

# AT&T syntax for x86
att = Assembler(triple="x86_64", preamble="")
att.asm("movq %rbx, %rax")
```

### With Unicorn emulation

```python
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB
from flintmc import Assembler

asm = Assembler.cortex_m7_dp()
uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
uc.mem_map(0, 0x1000)

code = asm("mov r0, #42; add r0, r0, #1")
uc.mem_write(0, code)
uc.emu_start(0 | 1, len(code))  # | 1 for Thumb mode
print(uc.reg_read(0))           # 43
```

### Shellcode / x86_64

```python
asm = Assembler.x86_64()

# Function prologue + epilogue
asm.asm("push rbp; mov rbp, rsp; sub rsp, 0x20")
asm.asm("leave; ret")

# SSE
asm.asm("movaps xmm0, xmm1; addps xmm0, xmm2")

# Syscall stub (Linux)
asm.asm("""
    mov rdi, 1
    lea rsi, [rip + msg]
    mov rdx, 13
    mov rax, 1
    syscall
""")
```

### AArch64

```python
a64 = Assembler.aarch64()

# Basics
a64("mov x0, #42; ret")
a64("add x0, x1, x2")
a64("sdiv x0, x1, x2")

# NEON / SIMD
a64("fadd v0.4s, v1.4s, v2.4s")
a64("fmul d0, d1, d2")
a64("fmadd d0, d1, d2, d3")
a64("dup v0.4s, v1.s[0]")

# Atomics (ARMv8.1-A LSE)
lse = Assembler(triple="aarch64", features="+lse")
lse("cas x0, x1, [x2]")
lse("ldadd x0, x1, [x2]")

# Function prologue/epilogue
a64.asm("""
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    ; ... body ...
    ldp x29, x30, [sp], #16
    ret
""")

# Conditional select
a64("cmp x0, #0; csel x1, x2, x3, eq")
```

### Cortex-M specifics

```python
arm = Assembler.cortex_m7_dp()

# Special registers (keystone can't assemble these)
arm("mrs r0, PRIMASK")
arm("msr BASEPRI, r0")
arm("mrs r0, FAULTMASK")
arm("msr CONTROL, r0")

# FPU — fused multiply-accumulate (also missing from keystone)
arm("vfma.f32 s0, s1, s2")
arm("vfms.f32 s0, s1, s2")
arm("vfnma.f32 s0, s1, s2")

# Integer divide
arm("udiv r0, r1, r2")
arm("sdiv r0, r1, r2")

# DSP / saturating arithmetic
arm("qadd r0, r1, r2")
arm("ssat r0, #16, r1")

# Bitfield operations
arm("bfi r0, r1, #8, #4")
arm("ubfx r0, r1, #0, #8")
```

### Enforcing FPU constraints

```python
# Single-precision only — rejects .f64 at assembly time
sp = Assembler.cortex_m7_sp()
sp.asm("vadd.f32 s0, s1, s2")   # ok
sp.asm("vadd.f64 d0, d1, d2")   # raises AsmError

# Double-precision enabled
dp = Assembler.cortex_m7_dp()
dp.asm("vadd.f64 d0, d1, d2")   # ok
```

### Error handling

```python
from flintmc import Assembler, AsmError

asm = Assembler.x86_64()
try:
    asm.asm("bad_instruction")
except AsmError as e:
    print(e)  # LLVM diagnostic with adjusted line numbers
```

Results are cached (LRU, thread-safe) — identical source strings return the same bytes without re-invoking llvm-mc.

## Install

```
brew install llvm     # macOS
apt install llvm      # Linux

pip install flintmc   # or: uv add flintmc
```

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

## LLVM compatibility

Tested with LLVM 22.x (Homebrew). Should work with LLVM 15+ — the llvm-mc command-line interface and ELF output format have been stable for years. If you hit an issue with an older LLVM version, file a bug.

## Testing

```
uv sync --group dev
uv run pytest                                       # core + x86 tests (1300+)

uv sync --group compat
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
uv run pytest tests/test_thumb2_vs_keystone.py      # keystone parity

uv run pytest tests/test_thumb2_vs_gas.py           # GAS ground-truth (if arm-none-eabi-as installed)
```

## Requirements

- Python >= 3.10
- LLVM (`llvm-mc` binary) — `brew install llvm` / `apt install llvm`
