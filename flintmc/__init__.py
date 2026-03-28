"""flintmc — LLVM-backed multi-architecture assembler for Python."""

__version__ = "0.2.1"

import logging

from .assembler import Assembler, AsmError, find_llvm_mc, set_libllvm_path, set_llvm_mc_path
from .common import UnsupportedArchitectureError
from .llvm_capi import register_arch_mapping

__all__ = ["Assembler", "AsmError", "UnsupportedArchitectureError", "find_llvm_mc", "set_libllvm_path", "set_llvm_mc_path", "register_arch_mapping", "asm", "default", "__version__"]

# Configure silent-by-default logger
logging.getLogger("flintmc").addHandler(logging.NullHandler())

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
