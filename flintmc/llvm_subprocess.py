import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .common import AsmError, _fix_error

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
        mc_path = (
            Path(env_path)
            / "bin"
            / ("llvm-mc.exe" if sys.platform == "win32" else "llvm-mc")
        )
        if mc_path.is_file():
            return str(mc_path)

    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/opt/llvm/bin/llvm-mc",  # Apple Silicon Homebrew
            "/usr/local/opt/llvm/bin/llvm-mc",  # Intel Homebrew
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            "/usr/bin/llvm-mc",
            "/usr/lib/llvm/bin/llvm-mc",
        ]
        # Versioned LLVM installs: /usr/bin/llvm-mc-18, etc.
        versioned = sorted(glob.glob("/usr/bin/llvm-mc-[0-9]*"), reverse=True)
        candidates.extend(versioned)

    for c in candidates:
        if Path(c).is_file():
            return c
    return shutil.which("llvm-mc")


class LlvmSubprocessBackend:
    """Assembly backend using the llvm-mc subprocess.

    Reliable fallback when libLLVM is not available. Each call spawns
    a new process, which is slower (~10ms) but provides strong isolation.
    """

    def __init__(self, triple: str, cpu: str = "", features: str = "", llvm_mc: str = ""):
        self.triple = triple
        self.cpu = cpu
        self.features = features
        self.llvm_mc = llvm_mc

        cmd = [llvm_mc, f"-triple={triple}", "-filetype=obj", "-o", "-"]
        if cpu:
            cmd.append(f"-mcpu={cpu}")
        if features:
            cmd.append(f"-mattr={features}")
        self._cmd = cmd

    def asm(self, source: str, preamble_lines: int = 0) -> bytes:
        """Assemble source text, return raw ELF bytes."""
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
