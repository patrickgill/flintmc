"""flintmc — Multi-architecture assembler backed by llvm-mc."""

from .assembler import Assembler, AsmError, find_llvm_mc

__all__ = ["Assembler", "AsmError", "find_llvm_mc", "asm", "default"]

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
