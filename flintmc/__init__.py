"""flintmc — LLVM-backed multi-architecture assembler for Python."""

__version__ = "0.2.0"

from .assembler import Assembler, AsmError, InstructionInfo
from .errors import UnsupportedArchitectureError
from .common import register_default_preamble
from .disassembler import Disassembler, DisasmInstruction
from .llvm_capi import register_arch_mapping, set_libllvm_path as _set_libllvm_path
from .llvm_subprocess import find_llvm_mc, set_llvm_mc_path


def set_libllvm_path(path: str) -> None:
    """Set the path to libLLVM, invalidating cached backend detection."""
    _set_libllvm_path(path)
    Assembler._resolved_default = None

__all__ = [
    "Assembler",
    "AsmError",
    "InstructionInfo",
    "Disassembler",
    "DisasmInstruction",
    "UnsupportedArchitectureError",
    "find_llvm_mc",
    "set_libllvm_path",
    "set_llvm_mc_path",
    "register_arch_mapping",
    "register_default_preamble",
    "asm",
    "default",
    "__version__",
]

default: Assembler | None = None


def asm(source: str) -> bytes:
    """Assemble using the module-level default assembler.

    Set ``flintmc.default`` first::

        import flintmc
        flintmc.default = flintmc.Assembler.x86_64()
        flintmc.asm("ret")  # b'\\xc3'

    Raises ``RuntimeError`` if no default has been configured.
    """
    if default is None:
        raise RuntimeError(
            "No default assembler configured. Set flintmc.default first:\n"
            "  import flintmc\n"
            "  flintmc.default = flintmc.Assembler.x86_64()"
        )
    return default.asm(source)
