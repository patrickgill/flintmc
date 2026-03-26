# Changelog

## 0.2.0

- LLVM C API backend via ctypes — default when `libLLVM` is available
- `LLVMContextSetDiagnosticHandler` intercepts assembly errors safely
- Thread-safe C API calls (lock-serialized)
- Subprocess fallback when `libLLVM` unavailable
- Architecture profiles: `armv6m`, `armv7m`, `armv8m`, `x86_32` (nyxstone parity)
- `backend=` parameter to force `"capi"` or `"subprocess"`
- Plain class replaces dataclass (mypy-compatible, `triple` is required)
- `__call__` shorthand: `asm("nop")` == `asm.asm("nop")`
- Module-level `flintmc.asm()` with configurable default
- Smart semicolon splitting (preserves quoted strings and comments)
- Bounded LRU cache (4096 entries, thread-safe)
- Error line-number adjustment for injected preamble
- Subprocess timeout (10s)
- `py.typed` marker

## 0.1.0

- Initial release
- Subprocess backend (`llvm-mc`)
- ARM Thumb-2 + x86/x86_64 + AArch64 support
- Cortex-M profiles: `cortex_m7_sp`, `cortex_m7_dp`, `cortex_m4`, `cortex_m33`, `cortex_m0`
- ELF `.text` extraction (32 + 64-bit)
- Auto-detected preamble per architecture
- Keystone parity tests (82 instructions)
- GAS ground-truth tests (118 instructions)
- 1300+ x86 encoding tests
