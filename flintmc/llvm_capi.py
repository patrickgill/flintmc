"""In-process LLVM assembly via the C API.

Uses ctypes to call libLLVM directly — no subprocess, no temp files.
Faster than the subprocess backend for cold calls as it avoids process
fork/exec overhead.

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
import ctypes.util
import glob
import logging
import os
import shutil
import sys
import threading
import weakref
from pathlib import Path
from typing import Any

from .common import AsmError, UnsupportedArchitectureError

logger = logging.getLogger("flintmc")

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
        "/usr/lib64/libLLVM.so",
    ],
    "win32": [
        "C:\\Program Files\\LLVM\\bin\\LLVM-C.dll",
        "C:\\Program Files (x86)\\LLVM\\bin\\LLVM-C.dll",
    ],
}

_lib: ctypes.CDLL | None = None
_initialized_arches: set[str] = set()
_init_lock = threading.Lock()
_active_backends = 0


def _load_llvm() -> ctypes.CDLL | None:
    """Try to load libLLVM. Returns None if not found.
    
    Order of preference:
    1. LLVM_PATH environment variable (root of an LLVM installation)
    2. Windows-specific PATH discovery
    3. Platform-standard installation paths (Homebrew, apt, etc.)
    """
    global _lib
    with _init_lock:
        if _lib is not None:
            return _lib

        platform = sys.platform
        candidates = []

        # 1. Check standard LLVM_PATH environment variable
        env_path = os.environ.get("LLVM_PATH")
        if env_path:
            p = Path(env_path)
            if platform == "win32":
                candidates.append(str(p / "bin" / "LLVM-C.dll"))
                candidates.append(str(p / "bin" / "libLLVM.dll"))
            else:
                ext = "dylib" if platform == "darwin" else "so"
                # Check both lib and lib64
                candidates.append(str(p / "lib" / f"libLLVM.{ext}"))
                candidates.append(str(p / "lib64" / f"libLLVM.{ext}"))

        if platform == "win32":
            candidates.extend(_LLVM_PATHS["win32"])
            
            # 2. Check PATH via shutil.which
            mc_path = shutil.which("llvm-mc")
            if mc_path:
                bin_dir = Path(mc_path).parent
                candidates.append(str(bin_dir / "LLVM-C.dll"))
                candidates.append(str(bin_dir / "libLLVM.dll"))
                
            # 3. Last resort: ctypes find_library
            found = ctypes.util.find_library("LLVM-C")
            if found:
                candidates.append(found)
        else:
            plat_key = "darwin" if platform == "darwin" else "linux"
            candidates.extend(_LLVM_PATHS.get(plat_key, []))

            if platform == "linux":
                # Versioned and multiarch paths for Linux
                for base in ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]:
                    # glob for libLLVM.so.N, libLLVM-N.so, etc.
                    candidates.extend(glob.glob(f"{base}/libLLVM*.so*"))

        for path in candidates:
            p = Path(path)
            if p.exists() or (platform == "win32" and not p.is_absolute()):
                try:
                    _lib = ctypes.CDLL(path)
                except OSError:
                    continue
                
                # Any error inside _declare_argtypes (like missing functions)
                # should bubble up as it indicates a broken/incompatible install.
                _declare_argtypes(_lib)
                return _lib
    return None


def set_libllvm_path(path: str) -> None:
    """Explicitly set the path to libLLVM and (re)load it.
    
    Warning: This clears the internal library cache. Existing backend
    instances will continue to use the library they were created with,
    but new instances will use the new path.
    """
    global _lib, _initialized_arches
    with _init_lock:
        _lib = ctypes.CDLL(path)
        _declare_argtypes(_lib)
        # Arches must be re-initialized for the new library
        _initialized_arches.clear()


def is_available() -> bool:
    """Check if libLLVM is available on the system."""
    return _load_llvm() is not None


def _declare_argtypes(lib: ctypes.CDLL) -> None:
    """Declare ALL function signatures up front.

    This is not optional. Without argtypes, ctypes assumes c_int for
    all arguments, truncating 64-bit pointers on LP64 platforms.
    """

    def _get(name: str) -> Any:
        fn = getattr(lib, name, None)
        if fn is None:
            raise RuntimeError(f"Required LLVM function '{name}' not found in library")
        return fn

    # Context
    _get("LLVMContextCreate").restype = VP
    _get("LLVMContextCreate").argtypes = []
    _get("LLVMContextDispose").restype = None
    _get("LLVMContextDispose").argtypes = [VP]
    _get("LLVMContextSetDiagnosticHandler").restype = None
    _get("LLVMContextSetDiagnosticHandler").argtypes = [VP, DIAG_HANDLER, ctypes.c_void_p]

    # Diagnostic info
    _get("LLVMGetDiagInfoDescription").restype = VP
    _get("LLVMGetDiagInfoDescription").argtypes = [VP]
    _get("LLVMGetDiagInfoSeverity").restype = ctypes.c_int
    _get("LLVMGetDiagInfoSeverity").argtypes = [VP]

    # Target lookup
    _get("LLVMGetTargetFromTriple").restype = BOOL
    _get("LLVMGetTargetFromTriple").argtypes = [CSTR, ctypes.POINTER(VP), ctypes.POINTER(CSTR)]

    # Target machine
    _get("LLVMCreateTargetMachine").restype = VP
    _get("LLVMCreateTargetMachine").argtypes = [VP, CSTR, CSTR, CSTR, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _get("LLVMDisposeTargetMachine").restype = None
    _get("LLVMDisposeTargetMachine").argtypes = [VP]

    # Data layout
    _get("LLVMCreateTargetDataLayout").restype = VP
    _get("LLVMCreateTargetDataLayout").argtypes = [VP]
    _get("LLVMSetModuleDataLayout").restype = None
    _get("LLVMSetModuleDataLayout").argtypes = [VP, VP]
    _get("LLVMDisposeTargetData").restype = None
    _get("LLVMDisposeTargetData").argtypes = [VP]

    # Module (context-aware)
    _get("LLVMModuleCreateWithNameInContext").restype = VP
    _get("LLVMModuleCreateWithNameInContext").argtypes = [CSTR, VP]
    _get("LLVMSetTarget").restype = None
    _get("LLVMSetTarget").argtypes = [VP, CSTR]
    _get("LLVMSetModuleInlineAsm2").restype = None
    _get("LLVMSetModuleInlineAsm2").argtypes = [VP, CSTR, SZ]
    _get("LLVMDisposeModule").restype = None
    _get("LLVMDisposeModule").argtypes = [VP]

    # Emit
    _get("LLVMTargetMachineEmitToMemoryBuffer").restype = BOOL
    _get("LLVMTargetMachineEmitToMemoryBuffer").argtypes = [
        VP, VP, ctypes.c_int, ctypes.POINTER(CSTR), ctypes.POINTER(VP)
    ]

    # Memory buffer
    _get("LLVMGetBufferStart").restype = ctypes.POINTER(ctypes.c_char)
    _get("LLVMGetBufferStart").argtypes = [VP]
    _get("LLVMGetBufferSize").restype = SZ
    _get("LLVMGetBufferSize").argtypes = [VP]
    _get("LLVMDisposeMemoryBuffer").restype = None
    _get("LLVMDisposeMemoryBuffer").argtypes = [VP]
    _get("LLVMDisposeMessage").restype = None
    _get("LLVMDisposeMessage").argtypes = [VP]


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
    "arm64": "AArch64",
    "riscv32": "RISCV",
    "riscv64": "RISCV",
    "wasm32": "WebAssembly",
    "wasm64": "WebAssembly",
    "ppc64": "PowerPC",
    "ppc64le": "PowerPC",
    "powerpc": "PowerPC",
    "sparc": "Sparc",
    "s390x": "SystemZ",
    "mips": "Mips",
    "loongarch64": "LoongArch",
}


def register_arch_mapping(prefix: str, llvm_name: str) -> None:
    """Register a new triple-prefix to LLVM architecture name mapping.
    
    Example:
        register_arch_mapping("mycpu", "MyArch")
        # Allows triples starting with 'mycpu' to resolve LLVMInitializeMyArch*
    """
    _TRIPLE_TO_ARCH[prefix.lower()] = llvm_name


def _arch_for_triple(triple: str) -> str | None:
    t = triple.lower()
    for prefix, arch in _TRIPLE_TO_ARCH.items():
        if t.startswith(prefix):
            return arch
    return None


def _init_target(lib: ctypes.CDLL, arch: str) -> None:
    with _init_lock:
        if arch in _initialized_arches:
            return

        for suffix in ["TargetInfo", "AsmParser", "AsmPrinter", "Target", "TargetMC"]:
            name = f"LLVMInitialize{arch}{suffix}"
            fn = getattr(lib, name, None)
            if fn is None:
                raise UnsupportedArchitectureError(
                    f"LLVM symbol '{name}' not found. "
                    f"Is {arch} support enabled in your LLVM build?"
                )
            fn.restype = None
            fn.argtypes = []
            fn()

        _initialized_arches.add(arch)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class LlvmCApiBackend:
    """In-process assembly backend using LLVM's C API.

    Creates a private LLVMContext with a diagnostic handler that captures
    errors instead of calling exit(). Reuses a single module and data
    layout for all calls to minimize overhead.

    NOT thread-safe on its own — intended to be used as a thread-local
    instance by the Assembler class.

    Cost: ~0.15ms per call (vs ~10ms for subprocess).
    """

    def __init__(self, lib: ctypes.CDLL, triple: str, cpu: str, features: str) -> None:
        global _active_backends
        self._lib = lib
        self._triple = triple.encode()
        self._finalizer = None

        # Per-backend error accumulator.
        self._errors: list[str] = []

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
            raise UnsupportedArchitectureError(f"LLVM target lookup failed for '{triple}': {msg}")

        self._tm = lib.LLVMCreateTargetMachine(
            target, self._triple, cpu.encode(), features.encode(), 0, 0, 0
        )
        if not self._tm:
            lib.LLVMContextDispose(self._ctx)
            raise UnsupportedArchitectureError(f"Failed to create TargetMachine for '{triple}'")

        # Performance fix: Reuse a single module for all calls. LLVMSetModuleInlineAsm2
        # overwrites existing assembly, avoiding the cost of per-call module creation.
        self._mod = lib.LLVMModuleCreateWithNameInContext(b"flintmc", self._ctx)
        if not self._mod:
            lib.LLVMDisposeTargetMachine(self._tm)
            lib.LLVMContextDispose(self._ctx)
            raise RuntimeError("LLVMModuleCreateWithNameInContext() failed to create module")
        lib.LLVMSetTarget(self._mod, self._triple)

        # Performance fix: Pre-create and set data layout once to avoid per-call setup cost.
        self._dl = lib.LLVMCreateTargetDataLayout(self._tm)
        if not self._dl:
            lib.LLVMDisposeTargetMachine(self._tm)
            lib.LLVMDisposeModule(self._mod)
            lib.LLVMContextDispose(self._ctx)
            raise RuntimeError("LLVMCreateTargetDataLayout() failed")
        lib.LLVMSetModuleDataLayout(self._mod, self._dl)

        with _init_lock:
            _active_backends += 1

        # Register finalizer for automatic cleanup.
        self._finalizer = weakref.finalize(
            self, _dispose_resources, lib, self._ctx, self._tm, self._mod, self._dl
        )

    def _on_diagnostic(self, info: int, _ctx: int) -> None:
        """LLVM diagnostic callback. Captures error messages.

        Called by LLVM from within emit.
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

        Raises RuntimeError with LLVM diagnostic on assembly errors.
        """
        lib = self._lib

        self._errors.clear()

        src = source.encode()
        # Injects assembly into the pre-configured module.
        lib.LLVMSetModuleInlineAsm2(self._mod, src, len(src))

        buf = VP()
        err = CSTR()
        # Emits the module (containing the inline asm) to a memory buffer.
        rc = lib.LLVMTargetMachineEmitToMemoryBuffer(
            self._tm, self._mod, LLVMObjectFile, ctypes.byref(err), ctypes.byref(buf)
        )

        # Diagnostics fire during emit, even when rc == 0
        captured = list(self._errors)
        self._errors.clear()

        if rc != 0:
            msg = err.value.decode() if err.value else "assembly failed"
            lib.LLVMDisposeMessage(err)
            # Combine return-code error with any captured diagnostics
            if captured:
                msg = "\n".join(captured) + "\n" + msg
            raise RuntimeError(msg)

        if captured:
            if buf.value:
                lib.LLVMDisposeMemoryBuffer(buf)
            raise RuntimeError("\n".join(captured))

        start = lib.LLVMGetBufferStart(buf)
        sz = lib.LLVMGetBufferSize(buf)
        elf = bytes(ctypes.cast(start, ctypes.POINTER(ctypes.c_char * sz)).contents)
        lib.LLVMDisposeMemoryBuffer(buf)
        return elf

    def close(self) -> None:
        """Release the TargetMachine and context."""
        if self._finalizer:
            self._finalizer()


def _dispose_resources(lib: Any, ctx: int, tm: int, mod: int, dl: int) -> None:
    """Standalone cleanup function for LlvmCApiBackend."""
    global _active_backends
    with _init_lock:
        _active_backends -= 1

    if tm:
        lib.LLVMDisposeTargetMachine(tm)
    if ctx:
        if mod:
            lib.LLVMDisposeModule(mod)
        if dl:
            lib.LLVMDisposeTargetData(dl)
        lib.LLVMContextDispose(ctx)


def try_create_backend(triple: str, cpu: str, features: str) -> LlvmCApiBackend | None:
    """Try to create a C API backend. Returns None if libLLVM unavailable."""
    lib = _load_llvm()
    if lib is None:
        return None

    arch = _arch_for_triple(triple)
    if arch is None:
        raise UnsupportedArchitectureError(f"Unsupported or unknown architecture for triple: {triple}")

    _init_target(lib, arch)

    # Propagate UnsupportedArchitectureError if target lookup/machine creation fails
    return LlvmCApiBackend(lib, triple, cpu, features)
