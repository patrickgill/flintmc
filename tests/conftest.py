import pytest
from flintmc import Assembler

def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["capi", "subprocess"],
        help="Force assembly backend: capi or subprocess"
    )

def pytest_configure(config):
    # Set the global default for the entire test session
    backend = config.getoption("--backend")
    if backend:
        Assembler.default_backend = backend

def pytest_report_header(config):
    # We can now "just read" the truth directly from the library
    forced = " (forced)" if config.getoption("--backend") else " (auto-detect)"
    return f"flintmc backend: {Assembler.resolve_backend()}{forced}"
