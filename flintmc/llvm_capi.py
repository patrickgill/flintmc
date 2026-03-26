"""In-process LLVM assembly via the C API.

Uses ctypes to call libLLVM directly — no subprocess, no temp files.
65x faster than the subprocess backend for cold calls.

The trick: LLVM's C API has no direct "assemble this string" function,
but LLVMSetModuleInlineAsm2() injects raw assembly into an LLVM IR
module, and LLVMTargetMachineEmitToMemoryBuffer() emits it as an ELF
object — going through the MC assembler layer internally.

Error handling: LLVM's inline asm path normally calls exit() on errors.
We intercept this by installing a LLVMContextSetDiagnosticHandler on a
private LLVMContext. The handler captures error diagnostics, and we
check for them after emit to raise AsmError instead of dying.

CRITICAL: All ctypes .argtypes MUST be declared before calling any
function. Without them, ctypes truncates 64-bit pointers to 32-bit on
ARM64 macOS, causing SIGSEGV. See research/llvm-capi-backend.py.
"""

import ctypes
import sys
import threading
from pathlib import Path

VP = ctypes.c_void_p
BOOL = ctypes.c_int
CSTR = ctypes.c_char_p
SZ = ctypes.c_size_t

LLVMObjectFile = 1

# Diagnostic handler callback type: void(LLVMDiagnosticInfoRef, void*)
DIAG_HANDLER = ctypes.CFUNCTYPE(None, VP, ctypes.c_void_p)

# LLVMDiagnosticSeverity enum
LLVMDSError = 0

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
    # Context
    lib.LLVMContextCreate.restype = VP
    lib.LLVMContextCreate.argtypes = []
    lib.LLVMContextDispose.restype = None
    lib.LLVMContextDispose.argtypes = [VP]
    lib.LLVMContextSetDiagnosticHandler.restype = None
    lib.LLVMContextSetDiagnosticHandler.argtypes = [VP, DIAG_HANDLER, ctypes.c_void_p]

    # Diagnostic info
    # Returns char* allocated with malloc — must be freed with LLVMDisposeMessage.
    # Use VP (not CSTR) to preserve the raw pointer for disposal.
    lib.LLVMGetDiagInfoDescription.restype = VP
    lib.LLVMGetDiagInfoDescription.argtypes = [VP]
    lib.LLVMGetDiagInfoSeverity.restype = ctypes.c_int
    lib.LLVMGetDiagInfoSeverity.argtypes = [VP]

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

    # Module (context-aware)
    lib.LLVMModuleCreateWithNameInContext.restype = VP
    lib.LLVMModuleCreateWithNameInContext.argtypes = [CSTR, VP]
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
    lib.LLVMDisposeMessage.argtypes = [VP]


# ---------------------------------------------------------------------------
# Target initialization
# ---------------------------------------------------------------------------

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
    t = triple.lower()
    for prefix, arch in _TRIPLE_TO_ARCH.items():
        if t.startswith(prefix):
            return arch
    return None


def _init_target(lib: ctypes.CDLL, arch: str) -> bool:
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

    Creates a private LLVMContext with a diagnostic handler that captures
    errors instead of calling exit(). Each asm() call creates a throwaway
    module in this context, injects inline asm, emits to ELF, and checks
    for captured diagnostics.

    Cost: ~0.2ms per cold call (vs ~10ms for subprocess).
    """

    def __init__(self, lib: ctypes.CDLL, triple: str, cpu: str, features: str) -> None:
        self._lib = lib
        self._triple = triple.encode()

        # Per-backend error accumulator + lock for thread safety.
        # The lock serializes ALL C API calls — LLVM contexts are not
        # thread-safe, and concurrent module creation/emit will segfault.
        self._errors: list[str] = []
        self._lock = threading.Lock()

        # Create private context with diagnostic handler.
        # The handler captures errors instead of letting LLVM call exit().
        # The callback must be stored as an attribute to prevent GC.
        self._diag_callback = DIAG_HANDLER(self._on_diagnostic)
        self._ctx = lib.LLVMContextCreate()
        lib.LLVMContextSetDiagnosticHandler(self._ctx, self._diag_callback, None)

        # Target machine (reused across calls)
        target = VP()
        err = CSTR()
        rc = lib.LLVMGetTargetFromTriple(self._triple, ctypes.byref(target), ctypes.byref(err))
        if rc != 0:
            msg = err.value.decode() if err.value else "unknown target"
            lib.LLVMDisposeMessage(err)
            lib.LLVMContextDispose(self._ctx)
            raise RuntimeError(f"LLVM target lookup failed for '{triple}': {msg}")

        self._tm = lib.LLVMCreateTargetMachine(
            target, self._triple, cpu.encode(), features.encode(), 0, 0, 0
        )
        if not self._tm:
            lib.LLVMContextDispose(self._ctx)
            raise RuntimeError(f"Failed to create TargetMachine for '{triple}'")

    def _on_diagnostic(self, info: int, _ctx: int) -> None:
        """LLVM diagnostic callback. Captures error messages.

        Called by LLVM from within emit — the lock is already held
        by asm(), so we don't re-acquire it here.
        """
        severity = self._lib.LLVMGetDiagInfoSeverity(info)
        if severity == LLVMDSError:
            desc_ptr = self._lib.LLVMGetDiagInfoDescription(info)
            if desc_ptr:
                desc_bytes = ctypes.cast(desc_ptr, CSTR).value
                msg = desc_bytes.decode() if desc_bytes else "unknown error"
                self._lib.LLVMDisposeMessage(desc_ptr)
            else:
                msg = "unknown error"
            self._errors.append(msg)

    def asm(self, source: str) -> bytes:
        """Assemble source, return raw ELF bytes. Caller extracts .text.

        Thread-safe — serializes all LLVM C API calls behind a lock.
        Raises RuntimeError with LLVM diagnostic on assembly errors.
        """
        lib = self._lib

        with self._lock:
            self._errors.clear()

            mod = lib.LLVMModuleCreateWithNameInContext(b"flintmc", self._ctx)
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

            # Diagnostics fire during emit, even when rc == 0
            captured = list(self._errors)
            self._errors.clear()

            if captured:
                if rc == 0 and buf.value:
                    lib.LLVMDisposeMemoryBuffer(buf)
                lib.LLVMDisposeModule(mod)
                raise RuntimeError("\n".join(captured))

            if rc != 0:
                msg = err.value.decode() if err.value else "assembly failed"
                lib.LLVMDisposeMessage(err)
                lib.LLVMDisposeModule(mod)
                raise RuntimeError(msg)

            start = lib.LLVMGetBufferStart(buf)
            sz = lib.LLVMGetBufferSize(buf)
            elf = bytes(ctypes.cast(start, ctypes.POINTER(ctypes.c_char * sz)).contents)
            lib.LLVMDisposeMemoryBuffer(buf)
            lib.LLVMDisposeModule(mod)
            return elf

    def close(self) -> None:
        """Release the TargetMachine and context."""
        if self._tm:
            self._lib.LLVMDisposeTargetMachine(self._tm)
            self._tm = None
        if self._ctx:
            self._lib.LLVMContextDispose(self._ctx)
            self._ctx = None

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
