"""In-process LLVM assembly via the C API.

Uses ctypes to call libLLVM directly — no subprocess, no temp files.
65x faster than the subprocess backend for cold calls.

The trick: LLVM's C API has no direct "assemble this string" function,
but LLVMSetModuleInlineAsm2() injects raw assembly into an LLVM IR
module, and LLVMTargetMachineEmitToMemoryBuffer() emits it as an ELF
object — going through the MC assembler layer internally.

CRITICAL: All ctypes .argtypes MUST be declared before calling any
function. Without them, ctypes truncates 64-bit pointers to 32-bit on
ARM64 macOS, causing SIGSEGV. See research/llvm-capi-backend.py for
the full crash analysis.
"""

import ctypes
import sys
from pathlib import Path

VP = ctypes.c_void_p
BOOL = ctypes.c_int
CSTR = ctypes.c_char_p
SZ = ctypes.c_size_t

LLVMObjectFile = 1

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LLVM_PATHS = {
    "darwin": [
        "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
        "/usr/local/opt/llvm/lib/libLLVM.dylib",
    ],
    "linux": [
        "/usr/lib/libLLVM.so",
        "/usr/lib/llvm-18/lib/libLLVM.so",
        "/usr/lib/llvm-17/lib/libLLVM.so",
        "/usr/lib/llvm-16/lib/libLLVM.so",
        "/usr/lib/llvm-15/lib/libLLVM.so",
    ],
}

_lib: ctypes.CDLL | None = None
_initialized_arches: set[str] = set()


def _load_llvm() -> ctypes.CDLL | None:
    """Try to load libLLVM. Returns None if not found."""
    global _lib
    if _lib is not None:
        return _lib

    platform = "darwin" if sys.platform == "darwin" else "linux"
    candidates = _LLVM_PATHS.get(platform, [])

    for path in candidates:
        if Path(path).exists():
            try:
                _lib = ctypes.CDLL(path)
                _declare_argtypes(_lib)
                return _lib
            except OSError:
                continue
    return None


def _declare_argtypes(lib: ctypes.CDLL) -> None:
    """Declare ALL function signatures up front.

    This is not optional. Without argtypes, ctypes assumes c_int for
    all arguments, truncating 64-bit pointers on LP64 platforms.
    """
    # Target init — set dynamically per arch in _init_target()

    # Target lookup
    lib.LLVMGetTargetFromTriple.restype = BOOL
    lib.LLVMGetTargetFromTriple.argtypes = [CSTR, ctypes.POINTER(VP), ctypes.POINTER(CSTR)]

    # Target machine
    lib.LLVMCreateTargetMachine.restype = VP
    lib.LLVMCreateTargetMachine.argtypes = [VP, CSTR, CSTR, CSTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.LLVMDisposeTargetMachine.restype = None
    lib.LLVMDisposeTargetMachine.argtypes = [VP]

    # Data layout
    lib.LLVMCreateTargetDataLayout.restype = VP
    lib.LLVMCreateTargetDataLayout.argtypes = [VP]
    lib.LLVMSetModuleDataLayout.restype = None
    lib.LLVMSetModuleDataLayout.argtypes = [VP, VP]

    # Module
    lib.LLVMModuleCreateWithName.restype = VP
    lib.LLVMModuleCreateWithName.argtypes = [CSTR]
    lib.LLVMSetTarget.restype = None
    lib.LLVMSetTarget.argtypes = [VP, CSTR]
    lib.LLVMSetModuleInlineAsm2.restype = None
    lib.LLVMSetModuleInlineAsm2.argtypes = [VP, CSTR, SZ]
    lib.LLVMDisposeModule.restype = None
    lib.LLVMDisposeModule.argtypes = [VP]

    # Emit
    lib.LLVMTargetMachineEmitToMemoryBuffer.restype = BOOL
    lib.LLVMTargetMachineEmitToMemoryBuffer.argtypes = [
        VP, VP, ctypes.c_int, ctypes.POINTER(CSTR), ctypes.POINTER(VP)
    ]

    # Memory buffer
    lib.LLVMGetBufferStart.restype = ctypes.POINTER(ctypes.c_char)
    lib.LLVMGetBufferStart.argtypes = [VP]
    lib.LLVMGetBufferSize.restype = SZ
    lib.LLVMGetBufferSize.argtypes = [VP]
    lib.LLVMDisposeMemoryBuffer.restype = None
    lib.LLVMDisposeMemoryBuffer.argtypes = [VP]
    lib.LLVMDisposeMessage.restype = None
    lib.LLVMDisposeMessage.argtypes = [CSTR]


# ---------------------------------------------------------------------------
# Target initialization
# ---------------------------------------------------------------------------

# Map triple prefixes to LLVM target names
_TRIPLE_TO_ARCH = {
    "thumb": "ARM",
    "arm": "ARM",
    "x86_64": "X86",
    "x86-64": "X86",
    "i686": "X86",
    "i386": "X86",
    "aarch64": "AArch64",
    "riscv32": "RISCV",
    "riscv64": "RISCV",
    "mips": "Mips",
}


def _arch_for_triple(triple: str) -> str | None:
    """Map an LLVM triple to the LLVM target architecture name."""
    t = triple.lower()
    for prefix, arch in _TRIPLE_TO_ARCH.items():
        if t.startswith(prefix):
            return arch
    return None


def _init_target(lib: ctypes.CDLL, arch: str) -> bool:
    """Initialize an LLVM target backend. Safe to call multiple times."""
    if arch in _initialized_arches:
        return True

    for suffix in ["TargetInfo", "Target", "AsmPrinter", "AsmParser", "TargetMC"]:
        name = f"LLVMInitialize{arch}{suffix}"
        fn = getattr(lib, name, None)
        if fn is None:
            return False
        fn.restype = None
        fn.argtypes = []
        fn()

    _initialized_arches.add(arch)
    return True


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class LlvmCApiBackend:
    """In-process assembly backend using LLVM's C API.

    Holds a reusable TargetMachine. Each asm() call creates a throwaway
    module, injects inline asm, emits to an ELF memory buffer, and
    extracts .text.

    Cost: ~0.2ms per cold call (vs ~10ms for subprocess).
    """

    def __init__(self, lib: ctypes.CDLL, triple: str, cpu: str, features: str) -> None:
        self._lib = lib
        self._triple = triple.encode()

        target = VP()
        err = CSTR()
        rc = lib.LLVMGetTargetFromTriple(self._triple, ctypes.byref(target), ctypes.byref(err))
        if rc != 0:
            msg = err.value.decode() if err.value else "unknown target"
            raise RuntimeError(f"LLVM target lookup failed for '{triple}': {msg}")

        self._tm = lib.LLVMCreateTargetMachine(
            target, self._triple, cpu.encode(), features.encode(), 0, 0, 0
        )
        if not self._tm:
            raise RuntimeError(f"Failed to create TargetMachine for '{triple}'")

    def asm(self, source: str) -> bytes:
        """Assemble source, return .text bytes."""
        lib = self._lib

        mod = lib.LLVMModuleCreateWithName(b"flintmc")
        lib.LLVMSetTarget(mod, self._triple)
        dl = lib.LLVMCreateTargetDataLayout(self._tm)
        lib.LLVMSetModuleDataLayout(mod, dl)

        src = source.encode()
        lib.LLVMSetModuleInlineAsm2(mod, src, len(src))

        buf = VP()
        err = CSTR()
        rc = lib.LLVMTargetMachineEmitToMemoryBuffer(
            self._tm, mod, LLVMObjectFile, ctypes.byref(err), ctypes.byref(buf)
        )

        if rc != 0:
            msg = err.value.decode() if err.value else "assembly failed"
            lib.LLVMDisposeModule(mod)
            raise RuntimeError(msg)

        start = lib.LLVMGetBufferStart(buf)
        sz = lib.LLVMGetBufferSize(buf)
        elf = bytes(ctypes.cast(start, ctypes.POINTER(ctypes.c_char * sz)).contents)
        lib.LLVMDisposeMemoryBuffer(buf)
        lib.LLVMDisposeModule(mod)
        return elf  # caller extracts .text

    def close(self) -> None:
        """Release the TargetMachine."""
        if self._tm:
            self._lib.LLVMDisposeTargetMachine(self._tm)
            self._tm = None

    def __del__(self) -> None:
        self.close()


def try_create_backend(triple: str, cpu: str, features: str) -> LlvmCApiBackend | None:
    """Try to create a C API backend. Returns None if libLLVM unavailable."""
    lib = _load_llvm()
    if lib is None:
        return None

    arch = _arch_for_triple(triple)
    if arch is None:
        return None

    if not _init_target(lib, arch):
        return None

    try:
        return LlvmCApiBackend(lib, triple, cpu, features)
    except RuntimeError:
        return None
