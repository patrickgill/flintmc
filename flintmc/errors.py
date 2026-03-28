"""Exception hierarchy for flintmc."""


class AsmError(Exception):
    """Assembly failed."""


class UnsupportedArchitectureError(AsmError):
    """Raised when the requested architecture is not supported."""
