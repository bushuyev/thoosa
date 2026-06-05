# ~/gdb-rust-helpers.py
#
# Load in gdb:
#   source ~/gdb-rust-helpers.py
#
# Usage:
#   rust-str EXPR
#   rust-location EXPR
#   rust-str-addr ADDRESS LENGTH
#
# Examples:
#   rust-str (*core.spawned_at).filename
#   rust-str (*core.spawned_at).filename.pointer
#   rust-location core.spawned_at
#   rust-location *core.spawned_at
#   rust-str-addr 0x614648abc123 47

import gdb
import re


def _eval(expr: str):
    return gdb.parse_and_eval(expr)


def _value_to_string(value) -> str:
    # Important: otherwise long Rust strings/arrays may be truncated.
    try:
        gdb.execute("set print elements 0", to_string=True)
    except Exception:
        pass

    return str(value)


def _decode_byte_array_from_gdb_text(text: str) -> str | None:
    """
    Handles GDB Rust output like:

        *const str [47, 104, 111, ...]

    or:

        NonNull<str> { pointer: *const str [47, 104, ...] }

    It extracts numeric byte arrays and UTF-8-decodes them.
    """

    candidates = []

    # Find bracketed lists containing only numbers, commas, whitespace.
    for match in re.finditer(r"\[([0-9,\s]+)\]", text, re.DOTALL):
        body = match.group(1)
        nums = re.findall(r"\d+", body)
        if nums:
            bs = bytes(int(x) for x in nums)
            candidates.append(bs)

    if not candidates:
        return None

    # Usually the Rust str byte array is the longest bracketed numeric list.
    bs = max(candidates, key=len)

    return bs.decode("utf-8", errors="replace")


def _try_field(value, field_name: str):
    try:
        return value[field_name]
    except Exception:
        return None


def _decode_rust_str_like(value) -> str:
    """
    Best-effort decoder for Rust string-like GDB values.

    Works well for:
      - core::ptr::non_null::NonNull<str>
      - NonNull<str>.pointer
      - *const str when GDB prints it as [bytes]
      - values whose GDB display contains a numeric byte array
    """

    # 1. Try direct textual representation.
    text = _value_to_string(value)
    decoded = _decode_byte_array_from_gdb_text(text)
    if decoded is not None:
        return decoded

    # 2. Try common Rust wrapper fields.
    for field in ("pointer", "ptr", "data_ptr", "data"):
        child = _try_field(value, field)
        if child is not None:
            text = _value_to_string(child)
            decoded = _decode_byte_array_from_gdb_text(text)
            if decoded is not None:
                return decoded

    # 3. Try dereferencing pointer/reference.
    try:
        deref = value.dereference()
        text = _value_to_string(deref)
        decoded = _decode_byte_array_from_gdb_text(text)
        if decoded is not None:
            return decoded
    except Exception:
        pass

    raise gdb.GdbError(
        "Could not decode as Rust string-like value. "
        "Try `p EXPR` first and check whether GDB prints a `[byte, byte, ...]` array. "
        "If you have raw address + length, use `rust-str-addr ADDRESS LENGTH`."
    )


class RustStr(gdb.Command):
    """
    Decode a Rust string-like expression.

    Usage:
      rust-str EXPR

    Examples:
      rust-str (*core.spawned_at).filename
      rust-str (*core.spawned_at).filename.pointer
    """

    def __init__(self):
        super().__init__("rust-str", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        expr = arg.strip()
        if not expr:
            raise gdb.GdbError("Usage: rust-str EXPR")

        value = _eval(expr)
        print(_decode_rust_str_like(value))


class RustStrAddr(gdb.Command):
    """
    Decode UTF-8 string from raw address and length.

    Usage:
      rust-str-addr ADDRESS LENGTH

    Example:
      rust-str-addr 0x614648abc123 47
    """

    def __init__(self):
        super().__init__("rust-str-addr", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        argv = gdb.string_to_argv(arg)
        if len(argv) != 2:
            raise gdb.GdbError("Usage: rust-str-addr ADDRESS LENGTH")

        addr = int(_eval(argv[0]))
        length = int(_eval(argv[1]))

        mem = gdb.selected_inferior().read_memory(addr, length)
        bs = bytes(mem)
        print(bs.decode("utf-8", errors="replace"))


def _maybe_deref_location(value):
    """
    Accept either Location or *Location.
    """

    t = value.type.strip_typedefs()

    if t.code == gdb.TYPE_CODE_PTR:
        return value.dereference()

    return value


class RustLocation(gdb.Command):
    """
    Print core::panic::Location-like values as:

      file:line:column

    Usage:
      rust-location EXPR

    Examples:
      rust-location core.spawned_at
      rust-location *core.spawned_at
      rust-location (*task_core).spawned_at
    """

    def __init__(self):
        super().__init__("rust-location", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        expr = arg.strip()
        if not expr:
            raise gdb.GdbError("Usage: rust-location EXPR")

        value = _eval(expr)
        loc = _maybe_deref_location(value)

        filename = _try_field(loc, "filename")
        line = _try_field(loc, "line")
        col = _try_field(loc, "col")

        if filename is None or line is None:
            raise gdb.GdbError(
                "Expression does not look like core::panic::Location. "
                "Expected fields: filename, line, col."
            )

        file_text = _decode_rust_str_like(filename)

        if col is not None:
            print(f"{file_text}:{int(line)}:{int(col)}")
        else:
            print(f"{file_text}:{int(line)}")


RustStr()
RustStrAddr()
RustLocation()

print("Loaded Rust GDB helpers: rust-str, rust-str-addr, rust-location")