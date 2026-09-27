"""Source preprocessing and error fixup utilities."""

import re

_STDIN_LINE_RE = re.compile(r"<(?:stdin|inline asm)>:(\d+):")

def _fix_error(stderr: str, preamble_lines: int) -> str:
    lines = []
    for line in stderr.splitlines():
        m = _STDIN_LINE_RE.match(line)
        if m:
            adjusted = max(1, int(m.group(1)) - preamble_lines)
            line = f"line {adjusted}:" + line[m.end() :]
        lines.append(line)
    return "\n".join(lines)

def _split_semicolons(source: str) -> str:
    if ";" not in source: return source
    out, in_block_comment = [], False
    for line in source.split("\n"):
        if ";" not in line and "/*" not in line and "*/" not in line:
            out.append(line)
            continue
        start, quote, i, n = 0, "", 0, len(line)
        while i < n:
            ch = line[i]
            if in_block_comment:
                if ch == "*" and i + 1 < n and line[i + 1] == "/": in_block_comment, i = False, i + 1
            elif quote:
                if ch == "\\": i += 1  # skip escaped char
                elif ch == quote: quote = ""
            elif ch in ('"', "'"): quote = ch
            else:
                if ch == "/" and i + 1 < n and line[i + 1] == "*": in_block_comment, i = True, i + 1
                elif ch == ";": out.append(line[start:i]); start = i + 1
                # @  -> ARM line comment (always)
                # // -> C-style line comment
                # #  -> x86 AT&T line comment, but only when NOT followed
                #       by a digit/sign (to avoid treating ARM/AArch64
                #       immediate prefixes like #42 as comments)
                elif ch == "@" or (ch == "/" and i + 1 < n and line[i + 1] == "/") or (ch == "#" and (i + 1 >= n or line[i + 1] not in "0123456789-+(")): break
            i += 1
        out.append(line[start:])
    return "\n".join(out)

# Preset profiles shared by Assembler and Disassembler: name -> (triple, cpu, features)
_PROFILES: dict[str, tuple[str, str, str]] = {
    "armv6m": ("armv6m-none-eabi", "", ""),
    "armv7m": ("armv7m-none-eabi", "", ""),
    "armv8m": ("armv8m.main-none-eabi", "", ""),
    "cortex_m0": ("thumbv6m-none-eabi", "cortex-m0", ""),
    "cortex_m4": ("thumbv7em-none-eabi", "cortex-m4", "+vfp4,-fp64,-fpregs64"),
    "cortex_m7_sp": ("thumbv7em-none-eabi", "cortex-m7", "+fp-armv8d16sp,-fp64,-fpregs64"),
    "cortex_m7_dp": ("thumbv7em-none-eabi", "cortex-m7", "+fp-armv8,+fp64"),
    "cortex_m33": ("thumbv8m.main-none-eabi", "cortex-m33", "+fp-armv8d16sp,+dsp,-fp64,-fpregs64"),
    "x86_64": ("x86_64", "", ""),
    "x86_32": ("i686", "", ""),
    "aarch64": ("aarch64", "", ""),
    "riscv64": ("riscv64", "generic-rv64", "+m,+a,+f,+d"),
    "riscv32": ("riscv32", "generic-rv32", "+m,+a,+f"),
    "loongarch64": ("loongarch64", "", ""),
    "avr": ("avr", "avr5", ""),
    "bpf": ("bpf", "", ""),
    "msp430": ("msp430", "", ""),
}

def _preset(cls, name, kw):
    """Construct *cls* from a named profile; ``cpu``/``features`` in *kw* override it."""
    triple, cpu, features = _PROFILES[name]
    return cls(triple, **{"cpu": cpu, "features": features, **kw})

# Triple components that select a non-ELF object format (Mach-O / COFF).
_NON_ELF = ("apple", "darwin", "macos", "ios", "tvos", "watchos", "macho",
            "windows", "win32", "msvc", "mingw", "cygwin", "coff")

def _elf_triple(triple: str) -> str:
    """Rewrite a Mach-O/COFF triple to ELF, keeping the arch.

    Mach-O leaves branches to local labels as unapplied relocations
    (``jmp end`` -> ``e9 00000000``), so everything is assembled as ELF.
    Machine code is identical; only the container differs.
    """
    arch, *rest = triple.split("-")
    if any(p.lower().startswith(_NON_ELF) for p in rest):
        return f"{arch}-unknown-none-elf"
    return triple

_PREAMBLE_OVERRIDES: dict[str, str] = {}

def register_default_preamble(prefix: str, preamble: str) -> None:
    """Register a default preamble for triples starting with *prefix*.

    Example::

        register_default_preamble("mycpu", ".option foo")
        Assembler(triple="mycpu-none-elf")  # preamble=".option foo"

    Overrides take precedence over the built-in rules.
    """
    _PREAMBLE_OVERRIDES[prefix.lower()] = preamble

def _default_preamble(triple: str) -> str:
    t = triple.lower()
    # User-registered overrides (longest prefix first)
    for prefix, preamble in sorted(_PREAMBLE_OVERRIDES.items(), key=lambda x: len(x[0]), reverse=True):
        if t.startswith(prefix):
            return preamble
    # Built-in rules
    # Check for Cortex-M triples: armv6m, armv7m, armv8m.main, etc.
    # Match "m" in the version part (after "armv"), not in "arm" itself.
    if t.startswith("thumb") or (t.startswith("armv") and "m" in t.split("-")[0][4:]): return ".syntax unified\n.thumb"
    if t.startswith("arm") and not t.startswith("arm64"): return ".syntax unified\n.arm"
    if "x86_64" in t or "x86-64" in t: return ".intel_syntax noprefix"
    if "i686" in t or "i386" in t: return ".intel_syntax noprefix\n.code32"
    return ""
