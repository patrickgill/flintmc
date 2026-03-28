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
