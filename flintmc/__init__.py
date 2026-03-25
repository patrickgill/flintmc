"""flintmc — Multi-architecture assembler backed by llvm-mc."""

from .assembler import Assembler, AsmError, find_llvm_mc

__all__ = ["Assembler", "AsmError", "find_llvm_mc"]
