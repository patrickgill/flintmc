# Subprocess Backend Refactor and Project Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the subprocess backend into its own module, consolidate shared utilities and exceptions into a common file, and update the project to version 0.2.1 while maintaining the existing public API.

**Architecture:** 
1. Move exceptions, ELF parsing, and string helpers to `flintmc/common.py`.
2. Create `flintmc/llvm_subprocess.py` to house the `LlvmSubprocessBackend` class.
3. Update `flintmc/llvm_capi.py` and `flintmc/assembler.py` to use these new locations.
4. Update `flintmc/__init__.py` and `pyproject.toml` for versioning.

**Tech Stack:** Python, subprocess, ctypes, pytest.

---

### Task 1: Create `flintmc/common.py` and Migrating Shared Logic

**Files:**
- Create: `flintmc/common.py`
- Modify: `flintmc/assembler.py` (read only for migration)

- [ ] **Step 1: Create `flintmc/common.py` with exceptions and helpers**

```python
import glob
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Optional

class AsmError(Exception):
    """Assembly failed."""

class UnsupportedArchitectureError(AsmError):
    """Raised when the requested architecture is not supported."""
    pass

_global_llvm_mc_path: Optional[str] = None

def set_llvm_mc_path(path: str) -> None:
    """Set a global path to the llvm-mc binary."""
    global _global_llvm_mc_path
    _global_llvm_mc_path = path

def find_llvm_mc() -> str | None:
    """Try to locate llvm-mc on the system."""
    if _global_llvm_mc_path:
        return _global_llvm_mc_path
        
    env_path = os.environ.get("LLVM_PATH")
    if env_path:
        mc_path = Path(env_path) / "bin" / ("llvm-mc.exe" if sys.platform == "win32" else "llvm-mc")
        if mc_path.is_file():
            return str(mc_path)

    candidates = []
    if sys.platform == "darwin":
        candidates = ["/opt/homebrew/opt/llvm/bin/llvm-mc", "/usr/local/opt/llvm/bin/llvm-mc"]
    elif sys.platform.startswith("linux"):
        candidates = ["/usr/bin/llvm-mc", "/usr/lib/llvm/bin/llvm-mc"]
        versioned = sorted(glob.glob("/usr/bin/llvm-mc-[0-9]*"), reverse=True)
        candidates.extend(versioned)

    for c in candidates:
        if Path(c).is_file():
            return c
    return shutil.which("llvm-mc")

_ELF_MAGIC = b"\x7fELF"

def _extract_text(elf: bytes) -> bytes:
    if len(elf) < 6 or elf[:4] != _ELF_MAGIC:
        raise AsmError("llvm-mc did not produce valid ELF output")
    ei_class = elf[4]
    if ei_class == 1: return _extract_text_elf32(elf)
    if ei_class == 2: return _extract_text_elf64(elf)
    raise AsmError(f"unsupported ELF class: {ei_class}")

def _extract_text_elf32(elf: bytes) -> bytes:
    e_shoff = struct.unpack_from("<I", elf, 0x20)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x2E)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x30)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x32)[0]
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, struct.Struct("<IIIIIIIIII"), 4, 5)

def _extract_text_elf64(elf: bytes) -> bytes:
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 0x3E)[0]
    return _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, struct.Struct("<IIQQQQIIQQ"), 4, 5)

def _find_text(elf, e_shoff, e_shentsize, e_shnum, e_shstrndx, shdr_struct, off_idx, size_idx):
    strtab_shdr = shdr_struct.unpack_from(elf, e_shoff + e_shstrndx * e_shentsize)
    strtab_data = elf[strtab_shdr[off_idx] : strtab_shdr[off_idx] + strtab_shdr[size_idx]]
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name_off = struct.unpack_from("<I", elf, off)[0]
        if strtab_data[name_off : name_off + 6] == b".text\x00":
            shdr = shdr_struct.unpack_from(elf, off)
            return elf[shdr[off_idx] : shdr[off_idx] + shdr[size_idx]]
    raise AsmError("no .text section in llvm-mc output")

_STDIN_LINE_RE = re.compile(r"<(?:stdin|inline asm)>:(\d+):")

def _fix_error(stderr: str, preamble_lines: int) -> str:
    lines = []
    for line in stderr.splitlines():
        m = _STDIN_LINE_RE.match(line)
        if m:
            adjusted = max(1, int(m.group(1)) - preamble_lines)
            line = f"line {adjusted}:" + line[m.end() :]
        lines.append(line)
    return "\n".join(lines)

def _split_semicolons(source: str) -> str:
    if ";" not in source: return source
    out, in_block_comment = [], False
    for line in source.split("\n"):
        if ";" not in line and "/*" not in line and "*/" not in line:
            out.append(line)
            continue
        start, in_quote, i, n = 0, False, 0, len(line)
        while i < n:
            ch = line[i]
            if in_block_comment:
                if ch == "*" and i + 1 < n and line[i + 1] == "/": in_block_comment, i = False, i + 1
            elif ch == '"': in_quote = not in_quote
            elif not in_quote:
                if ch == "/" and i + 1 < n and line[i + 1] == "*": in_block_comment, i = True, i + 1
                elif ch == ";": out.append(line[start:i]); start = i + 1
                elif ch in ("@", "#") or (ch == "/" and i + 1 < n and line[i + 1] == "/"): break
            i += 1
        out.append(line[start:])
    return "\n".join(out)

def _default_preamble(triple: str) -> str:
    t = triple.lower()
    if t.startswith("thumb") or (t.startswith("armv") and "m" in t.split("-")[0]): return ".syntax unified\n.thumb"
    if t.startswith("arm"): return ".syntax unified\n.arm"
    if "x86_64" in t or "x86-64" in t: return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t: return ".intel_syntax noprefix\n.code32"
    return ""
```

- [ ] **Step 2: Commit `common.py`**

```bash
git add flintmc/common.py
git commit -m "refactor: create common.py with shared utilities and exceptions"
```

### Task 2: Refactor `flintmc/llvm_capi.py`

**Files:**
- Modify: `flintmc/llvm_capi.py`

- [ ] **Step 1: Update imports and remove duplicate code**

```python
# Replace local AsmError and UnsupportedArchitectureError with imports
from .common import AsmError, UnsupportedArchitectureError
```

- [ ] **Step 2: Run tests to verify C API still works**

Run: `pytest tests/test_errors.py -v` (assuming C API is default)

- [ ] **Step 3: Commit**

```bash
git add flintmc/llvm_capi.py
git commit -m "refactor: use common.py exceptions in llvm_capi.py"
```

### Task 3: Create `flintmc/llvm_subprocess.py`

**Files:**
- Create: `flintmc/llvm_subprocess.py`

- [ ] **Step 1: Implement `LlvmSubprocessBackend`**

```python
import subprocess
from .common import AsmError, _fix_error

class LlvmSubprocessBackend:
    def __init__(self, triple: str, cpu: str = "", features: str = "", llvm_mc: str = ""):
        self.triple = triple
        self.cpu = cpu
        self.features = features
        self.llvm_mc = llvm_mc
        
        cmd = [llvm_mc, f"-triple={triple}", "-filetype=obj", "-o", "-"]
        if cpu: cmd.append(f"-mcpu={cpu}")
        if features: cmd.append(f"-mattr={features}")
        self._cmd = cmd

    def asm(self, source: str, preamble_lines: int = 0) -> bytes:
        try:
            result = subprocess.run(
                self._cmd,
                input=source.encode(),
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise AsmError("llvm-mc timed out (10s limit)")

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise AsmError(_fix_error(stderr, preamble_lines))

        return result.stdout
```

- [ ] **Step 2: Commit**

```bash
git add flintmc/llvm_subprocess.py
git commit -m "feat: add LlvmSubprocessBackend in llvm_subprocess.py"
```

### Task 4: Refactor `flintmc/assembler.py`

**Files:**
- Modify: `flintmc/assembler.py`

- [ ] **Step 1: Simplify `Assembler` and update imports**

Remove ELF parsing, string helpers, and subprocess logic. Import them from `common.py` and `llvm_subprocess.py`.

```python
from .common import (
    AsmError,
    UnsupportedArchitectureError,
    find_llvm_mc,
    set_llvm_mc_path,
    _extract_text,
    _fix_error,
    _split_semicolons,
    _default_preamble,
)
from .llvm_subprocess import LlvmSubprocessBackend
```

- [ ] **Step 2: Update `Assembler._run_subprocess` to use the new backend**

```python
    def _run_subprocess(self, source: str) -> bytes:
        if not hasattr(self._local, "subprocess"):
            self._local.subprocess = LlvmSubprocessBackend(
                self.triple, self.cpu, self.features, self.llvm_mc
            )
        elf = self._local.subprocess.asm(source, self._preamble_lines)
        return elf # Assembler.asm will call _extract_text
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 4: Commit**

```bash
git add flintmc/assembler.py
git commit -m "refactor: simplify assembler.py and use new backends"
```

### Task 5: Version Bump and Final Exports

**Files:**
- Modify: `flintmc/__init__.py`, `pyproject.toml`

- [ ] **Step 1: Update version and re-exports in `flintmc/__init__.py`**

```python
__version__ = "0.2.1"
from .assembler import Assembler, AsmError, find_llvm_mc, set_libllvm_path, set_llvm_mc_path
from .common import UnsupportedArchitectureError
# ... (rest of file)
```

- [ ] **Step 2: Update `pyproject.toml` version**

- [ ] **Step 3: Run final verification**

Run: `pytest tests/ --backend=subprocess -v`
Run: `pytest tests/ --backend=capi -v`

- [ ] **Step 4: Commit**

```bash
git add flintmc/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.2.1"
```
