"""Stress test for thread-safety and concurrency in flintmc.

This test spawns many threads to perform simultaneous assembly calls
using the same Assembler instance. If the locking or resource management
is broken, this should trigger a SIGSEGV or incorrect assembly results.
"""

import concurrent.futures
import pytest
from flintmc import Assembler

def test_extreme_concurrency():
    """Spawn 100 threads to assemble different instructions simultaneously."""
    asm = Assembler.x86_64()
    
    # Mix of instructions to ensure different outputs and cache keys
    # Some include semicolons to test the optimized _split_semicolons
    work_items = []
    for i in range(200):
        val = i % 255
        instr = f"mov eax, {val}; ret"
        expected_suffix = bytes([val, 0x00, 0x00, 0x00, 0xc3]) # mov eax, imm32; ret
        work_items.append((instr, expected_suffix))

    def worker(item):
        instr, expected_suffix = item
        res = asm.asm(instr)
        # Check that we got a valid result and it ends with our expected bytes
        # (mov eax, imm32 is 5 bytes, ret is 1 byte = 6 bytes total)
        assert len(res) == 6
        assert res.endswith(expected_suffix)
        return True

    # Use a large pool to force context switching and race conditions
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(worker, item) for item in work_items]
        results = [f.result() for f in futures]
    
    assert all(results)

def test_concurrent_error_handling():
    """Ensure error messages from one thread don't leak into another."""
    asm = Assembler.x86_64()
    
    def valid_worker():
        for _ in range(50):
            assert len(asm.asm("nop")) == 1
            
    def invalid_worker():
        for _ in range(50):
            try:
                asm.asm("not_a_real_instruction")
            except Exception as e:
                # Ensure the error message is relevant to this thread
                assert "not_a_real_instruction" in str(e).lower()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        f1 = [executor.submit(valid_worker) for _ in range(5)]
        f2 = [executor.submit(invalid_worker) for _ in range(5)]
        concurrent.futures.wait(f1 + f2)
        # Check for exceptions in futures
        for f in f1 + f2:
            f.result()

def test_cache_trashing_concurrency():
    """Spawn many threads, each assembling a unique instruction to trash the cache."""
    asm = Assembler.x86_64()
    # Cache maxsize is 4096. Let's push 8000 unique items.
    def worker(i):
        # Unique instruction for each i
        instr = f"mov eax, {i}"
        return asm.asm(instr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(worker, i) for i in range(8000)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert asm.cache_size == 4096

def test_concurrent_arch_init():
    """Create multiple Assemblers for different architectures simultaneously."""
    arches = ["x86_64", "aarch64", "thumbv7m-none-eabi", "riscv64"]
    
    def worker(triple):
        # This will trigger first-time initialization for the arch
        asm = Assembler(triple=triple)
        return len(asm.asm("nop")) > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, arch) for arch in arches]
        results = [f.result() for f in futures]
    
    assert all(results)
