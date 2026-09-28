"""GNU binutils backend for targets LLVM doesn't have (e.g. V850).

Assembles with ``<arch>-elf-as`` and disassembles with ``<arch>-elf-objdump``.
Tools are looked up in ``$BINUTILS_PATH/bin``, then on ``PATH``.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .errors import AsmError
from .common import _fix_error

# "   4:\t40 0a \tmov\t0, r1"  (address, hex bytes, text)
_LINE_RE = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*\t?(.*)$")

# V850 PC-relative branches: objdump prints the absolute target, but gas
# (like LLVM's disassembler) takes a displacement.  Not bsh/bsw (byte swap).
_V850_BRANCH_RE = re.compile(
    r"^(jr|jarl|b(?:gt|ge|lt|le|h|nh|l|nl|e|ne|c|nc|v|nv|n|p|r|sa|z|nz))\s+(0x[0-9a-f]+)\b")


# Optional ops gas accepts only with -mextension; objdump always decodes them.
_V850_EXTENSION = frozenset(
    "divhn divhun divn divun sdivhn sdivhun sdivn sdivun ldacc macacc macuacc stacch staccl".split())


def _v850_text(text: str, pc: int, extension: bool) -> str:
    """Rewrite objdump's V850 text into something gas reassembles."""
    if not extension and text.split(" ", 1)[0] in _V850_EXTENSION:
        return "(bad)"
    text = _V850_BRANCH_RE.sub(lambda b: _to_disp(b, pc), text)
    # "eipc/vip/mpm", "nc/nl": objdump lists every alias; gas wants one
    text = re.sub(r"\b(\w+)(?:/\w+)+", r"\1", text)
    # "dispose 16, {}, r0": r0 means no jump, and gas rejects it spelled out
    return re.sub(r"^(dispose .*\}), r0$", r"\1", text)


def _to_disp(m: re.Match, pc: int) -> str:
    d = (int(m.group(2), 16) - pc + 2**31) % 2**32 - 2**31  # signed 32-bit
    return f"{m.group(1)} {'-' if d < 0 else ''}{abs(d):#x}"


def find_tool(triple: str, tool: str) -> str | None:
    """Find ``<arch>-elf-<tool>`` (or ``<triple>-<tool>``) for *triple*."""
    arch = triple.split("-")[0]
    names = dict.fromkeys([f"{triple}-{tool}", f"{arch}-elf-{tool}"])
    env = os.environ.get("BINUTILS_PATH")
    for name in names:
        if env and (p := Path(env) / "bin" / name).is_file():
            return str(p)
        if p := shutil.which(name):
            return p
    return None


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10, **kw)
    except subprocess.TimeoutExpired:
        raise AsmError(f"{Path(cmd[0]).name} timed out (10s limit)") from None


class BinutilsBackend:
    """Assembly via GNU ``as`` (one subprocess per call, like llvm-mc)."""

    def __init__(self, triple: str, cpu: str = "", features: str = ""):
        gas = find_tool(triple, "as")
        if not gas:
            raise FileNotFoundError(f"GNU as not found for {triple} (set BINUTILS_PATH or PATH)")
        # ponytail: features are passed through as raw extra gas flags
        self._cmd = [gas, *([f"-m{cpu}"] if cpu else []), *filter(None, features.split(","))]

    def asm(self, source: str, preamble_lines: int = 0) -> bytes:
        """Assemble source text, return raw ELF bytes."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "a.o")
            r = _run([*self._cmd, "-o", out], input=source.encode())
            if r.returncode != 0:
                raise AsmError(_fix_error(r.stderr.decode(errors="replace").strip(), preamble_lines))
            return Path(out).read_bytes()


def disasm(objdump: str, mach: str, code: bytes, address: int,
           extension: bool = False) -> list[tuple[int, bytes, str]]:
    """Disassemble *code* with objdump, return ``(offset, bytes, text)`` per instruction."""
    if not code:
        return []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "code.bin")
        Path(path).write_bytes(code)
        r = _run([objdump, "-D", "-z", "-b", "binary", "-m", mach,
                  "--insn-width=16", f"--adjust-vma={address:#x}", path])
    if r.returncode != 0:
        raise AsmError(r.stderr.decode(errors="replace").strip())
    out = []
    for line in r.stdout.decode(errors="replace").splitlines():
        if m := _LINE_RE.match(line):
            pc = int(m.group(1), 16)
            text = re.sub(r"\s*<[^>]*>", "", m.group(3)).replace("\t", " ").strip()
            if mach.startswith("v850"):
                text = _v850_text(text, pc, extension)
            out.append((pc - address, bytes.fromhex(m.group(2)), text))
    return out
