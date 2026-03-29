# flintmc

LLVM-based multi-architecture assembler and disassembler for Python. Drop-in replacement for keystone-engine.

Calls directly into the LLVM already installed on your system. When you update LLVM, flintmc picks up new instruction support automatically.

Supports any target your LLVM supports: x86, x86_64, ARM Thumb-2, AArch64, RISC-V, AVR, BPF, MSP430, LoongArch, etc. Pure Python via ctypes, no vendored code, no compiled extensions.

## Install

```
brew install llvm     # macOS
apt install llvm      # Linux

pip install flintmc   # or: uv add flintmc
```

## Examples

### Assembly basics

```python
from flintmc import Assembler

asm = Assembler.x86_64()
asm("mov rax, rbx; ret")          # b'\x48\x89\xd8\xc3'
asm("syscall")                    # b'\x0f\x05'

asm = Assembler.cortex_m7_dp()
asm("mrs r0, PRIMASK")            # b'\xef\xf3\x10\x80'
asm("vfma.f32 s0, s1, s2")       # b'\xa0\xee\x81\x0a'

asm = Assembler.aarch64()
asm("mov x0, #42; ret")
```

### Disassembly

```python
from flintmc import Disassembler

dis = Disassembler.x86_64()
for instr in dis(b'\x48\x89\xd8\xc3'):
    print(f"{instr.offset:#x}: {instr.text}")
# 0x0: mov rax, rbx
# 0x3: ret
```

### Symbols and address

```python
asm = Assembler.cortex_m7_dp()

# External symbols are injected as .set directives
asm("bl handler", symbols={"handler": 0x80})

# Base address for PC-relative instructions
asm("adr r0, label", address=0x1000, symbols={"label": 0x1040})
```

### Per-instruction info

```python
asm = Assembler.x86_64()
for instr in asm.asm_each("nop; mov rax, rbx; ret"):
    print(f"{instr.offset:#x} [{instr.size}] {instr.source}")
# 0x0 [1] nop
# 0x1 [3] mov rax, rbx
# 0x4 [1] ret
```

### Multi-instruction, labels, directives

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

### Shellcode generation

```python
x64 = Assembler.x86_64()
payload = x64("""
    xor rdi, rdi
    mov al, 0x3c
    syscall
""")
```

### Hot-patching ARM firmware

```python
patcher = Assembler.cortex_m7_dp()
hook_addr = 0x20001000
patch = patcher(f"ldr pc, ={hook_addr}\n.ltorg")
```

### Bootloader data directives

```python
asm = Assembler.x86_32()
gdt_entry = asm("""
    .word 0xFFFF    # Limit
    .word 0x0000    # Base (low)
    .byte 0x00      # Base (mid)
    .byte 0x9A      # Access (exec/read)
    .byte 0xCF      # Granularity
    .byte 0x00      # Base (high)
""")
```

### Round-trip verification

`verify=True` disassembles the output and re-assembles it, checking that the bytes match. Catches encoding ambiguities and subtle misassembly.

```python
asm = Assembler.x86_64()
asm("nop; ret", verify=True)          # passes — round-trips cleanly

asm(".byte 0x48, 0x00", verify=True)  # AsmError: cannot disassemble output
```

Requires the C API backend (libLLVM) since disassembly is C API only. Only use with instructions — data directives (`.byte`, `.word`), literal pools (`.ltorg`), and alignment (`.align`) won't round-trip.

### Custom targets

```python
asm = Assembler(triple="riscv64", cpu="generic-rv64", features="+m,+a,+f,+d")
asm("addi x1, x0, 42")

# AT&T syntax for x86
att = Assembler(triple="x86_64", preamble="")
att("movq %rbx, %rax")
```

### Keystone migration

```python
# Before (Keystone):
# from keystone import KS_ARCH_X86, KS_MODE_64, Ks
# ks = Ks(KS_ARCH_X86, KS_MODE_64)
# code, _ = ks.asm("mov rax, 1")

# After (flintmc):
from flintmc import Assembler
asm = Assembler.x86_64()
code = asm("mov rax, 1")
```

## API

### Assembler

```python
Assembler(triple, *, cpu="", features="", preamble=None, backend=None)
```

- **`asm(source, *, address=None, symbols=None, verify=False)`** — assemble to bytes
- **`asm_each(source)`** — assemble and return a list of `InstructionInfo(offset, size, code, source)`
- **`close()`** — release backend resources for the calling thread
- Callable: `asm("nop")` is the same as `asm.asm("nop")`

`assemble` and `assemble_each` are aliases for `asm` and `asm_each`.

### Disassembler

```python
Disassembler(triple, *, cpu="", features="", intel=True)
```

- **`disasm(code, *, address=0)`** — disassemble to a list of `DisasmInstruction(offset, size, code, text)`
- **`available()`** — static method, checks if libLLVM is present
- **`close()`** — release resources for the calling thread
- Callable: `dis(code)` is the same as `dis.disasm(code)`

`disassemble` is an alias for `disasm`.

Requires the C API backend (libLLVM). No subprocess fallback.

### Context managers

```python
with Assembler.x86_64() as asm:
    code = asm("nop")

with Disassembler.x86_64() as dis:
    instrs = dis(code)
```

### Profiles

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

# RISC-V
Assembler.riscv64()            # RV64IMAFD
Assembler.riscv32()            # RV32IMAF

# Other
Assembler.avr()                # Atmel AVR (8-bit)
Assembler.bpf()                # eBPF
Assembler.msp430()             # TI MSP430 (16-bit)
Assembler.loongarch64()        # LoongArch 64-bit
```

### Error handling

```python
from flintmc import AsmError

try:
    asm("bad_instruction")
except AsmError as e:
    print(e)  # LLVM diagnostic with adjusted line numbers
```

### Backends

**C API (default):** Assembly source is injected into an LLVM module via `LLVMSetModuleInlineAsm2()`, emitted as an ELF object via `LLVMTargetMachineEmitToMemoryBuffer()`, and the `.text` section is extracted in pure Python. All via ctypes into your system's `libLLVM`. A `LLVMContextSetDiagnosticHandler` intercepts assembly errors so they become `AsmError` exceptions.

**Subprocess (fallback):** If `libLLVM` isn't found, falls back to piping through `llvm-mc -filetype=obj -o -`.

Force a backend with `backend="capi"` or `backend="subprocess"`.

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
uv run pytest                                       # 5100+ tests

uv sync --group compat
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
uv run pytest tests/test_thumb2_vs_keystone.py      # keystone parity

uv run pytest tests/test_thumb2_vs_gas.py           # GAS ground-truth (needs arm-none-eabi-as)
```

## Requirements

- Python >= 3.10
- LLVM 7.0+ (`libLLVM` shared library + `llvm-mc` binary)
