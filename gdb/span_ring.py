# source /home/bu/bin/gdb/span_active_ring.py
# span-active-stats
# continue
#
# After stall:
#
# Ctrl-C
# span-active-stats
# span-active-list 100
# span-active-dump 1000 /home/bu/tmp/span_active_ring.txt

import os
import shlex
import gdb
from collections import deque, defaultdict


TRACING_INSTRUMENT_POLL_SPEC = (
    "/home/bu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/"
    "tracing-0.1.43/src/instrument.rs:318"
)

TOKIO_CORE_POLL_BEGIN_SPEC = (
    "/home/bu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/"
    "tokio-1.52.3/src/runtime/task/core.rs:364"
)

TOKIO_CORE_POLL_RESULT_SPEC = (
    "/home/bu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/"
    "tokio-1.52.3/src/runtime/task/core.rs:379"
)

# Set to True if TOKIO_CORE_POLL_RESULT_SPEC points directly inside Poll::Pending branch.
CORE_RESULT_LINE_IS_ALREADY_PENDING_BRANCH = False

RING_MAX = 2048
MAX_STACK_DEPTH_PER_THREAD = 8

DELETE_EXISTING_MATCHING_BPS_ON_INSTALL = True


try:
    span_active_ring
except NameError:
    span_active_ring = deque(maxlen=RING_MAX)
else:
    if span_active_ring.maxlen != RING_MAX:
        span_active_ring = deque(span_active_ring, maxlen=RING_MAX)

try:
    active_core_stack_by_thread
except NameError:
    active_core_stack_by_thread = defaultdict(list)

try:
    span_active_breakpoints
except NameError:
    span_active_breakpoints = []

try:
    span_seq
except NameError:
    span_seq = 0

try:
    core_seq
except NameError:
    core_seq = 0


def _restore_param(name, value):
    if isinstance(value, bool):
        gdb.execute(f"set {name} {'on' if value else 'off'}", to_string=True)
    elif value is None:
        gdb.execute(f"set {name} unlimited", to_string=True)
    else:
        gdb.execute(f"set {name} {value}", to_string=True)


def _thread_key():
    th = gdb.selected_thread()
    if th is None:
        return ("no-thread",)
    return th.ptid


def _thread_label():
    th = gdb.selected_thread()
    if th is None:
        return "no-thread"
    return f"gdb_thread={th.num}, ptid={th.ptid}"


def _eval_first(exprs):
    last = None
    for expr in exprs:
        try:
            return gdb.parse_and_eval(expr)
        except Exception as ex:
            last = ex
    raise RuntimeError(f"all expressions failed; last={last}")


def _eval_tiny_str(exprs):
    # Use only for small scalar/string fields.
    try:
        return str(_eval_first(exprs))
    except Exception as ex:
        return f"<unavailable: {ex}>"


def _ptr_hex_first(exprs):
    # Do not stringify complex Rust values.
    ulong = gdb.lookup_type("unsigned long")
    last = None

    for expr in exprs:
        try:
            v = gdb.parse_and_eval(expr)
            return "0x%x" % int(v.cast(ulong))
        except Exception as ex:
            last = ex

    return f"<unavailable: {last}>"


def _core_ptr():
    return _ptr_hex_first([
        "self",
        "&self",
        "ptr",
    ])


def _span_file_line():
    file_s = _eval_tiny_str([
        "(*self.pointer).span.meta.0.file",
        "(*self.pointer).span.meta.value.file",
        "(*self.pointer).span.metadata.file",
    ])

    line_s = _eval_tiny_str([
        "(*self.pointer).span.meta.0.line",
        "(*self.pointer).span.meta.value.line",
        "(*self.pointer).span.metadata.line",
    ])

    return file_s, line_s


def _res_is_pending():
    if CORE_RESULT_LINE_IS_ALREADY_PENDING_BRANCH:
        return True

    old_depth = gdb.parameter("print max-depth")
    old_elements = gdb.parameter("print elements")
    old_pretty = gdb.parameter("print pretty")

    try:
        # Shallow enum print only. Avoid expanding Poll::Ready payload.
        gdb.execute("set print max-depth 1", to_string=True)
        gdb.execute("set print elements 0", to_string=True)
        gdb.execute("set print pretty off", to_string=True)

        s = str(gdb.parse_and_eval("res"))
        return "Pending" in s and "Ready" not in s

    except Exception:
        return False

    finally:
        _restore_param("print max-depth", old_depth)
        _restore_param("print elements", old_elements)
        _restore_param("print pretty", old_pretty)


def core_begin_capture():
    k = _thread_key()
    stack = active_core_stack_by_thread[k]

    # Defensive cleanup if result breakpoint was missed.
    if len(stack) >= MAX_STACK_DEPTH_PER_THREAD:
        del stack[:-1]

    stack.append({
        "thread": _thread_label(),
        "core_ptr": _core_ptr(),
        "last_span": None,
    })


def span_capture_if_active():
    global span_seq

    k = _thread_key()
    stack = active_core_stack_by_thread.get(k)

    # Hot path: if this Instrumented::poll is not inside a Core::poll interval,
    # return immediately without reading span fields.
    if not stack:
        return

    file_s, line_s = _span_file_line()

    span_seq += 1
    stack[-1]["last_span"] = {
        "seq": span_seq,
        "file": file_s,
        "line": line_s,
    }


def core_result_capture():
    global core_seq

    k = _thread_key()
    stack = active_core_stack_by_thread.get(k)

    if not stack:
        return

    ctx = stack.pop()

    try:
        if _res_is_pending():
            core_seq += 1
            span_active_ring.append({
                "seq": core_seq,
                "thread": ctx["thread"],
                "core_ptr": ctx["core_ptr"],
                "last_span": ctx["last_span"],
            })
    finally:
        if not stack:
            active_core_stack_by_thread.pop(k, None)


class InstrumentedPollBreakpoint(gdb.Breakpoint):
    def stop(self):
        try:
            span_capture_if_active()
        except Exception as ex:
            gdb.write(f"\n[span_capture_if_active failed: {ex}]\n")
        return False


class CoreBeginBreakpoint(gdb.Breakpoint):
    def stop(self):
        try:
            core_begin_capture()
        except Exception as ex:
            gdb.write(f"\n[core_begin_capture failed: {ex}]\n")
        return False


class CoreResultBreakpoint(gdb.Breakpoint):
    def stop(self):
        try:
            core_result_capture()
        except Exception as ex:
            gdb.write(f"\n[core_result_capture failed: {ex}]\n")
        return False


def _location_matches(spec, loc):
    if not loc:
        return False
    if spec in loc:
        return True
    return spec.split("/")[-1] in loc


def _delete_matching_existing_breakpoints():
    if not DELETE_EXISTING_MATCHING_BPS_ON_INSTALL:
        return

    specs = [
        TRACING_INSTRUMENT_POLL_SPEC,
        TOKIO_CORE_POLL_BEGIN_SPEC,
        TOKIO_CORE_POLL_RESULT_SPEC,
    ]

    for bp in gdb.breakpoints() or []:
        try:
            loc = getattr(bp, "location", "") or ""
            if any(_location_matches(spec, loc) for spec in specs):
                bp.delete()
        except Exception:
            pass


def span_active_install():
    global span_active_breakpoints

    for bp in list(span_active_breakpoints):
        try:
            if bp.is_valid():
                bp.delete()
        except Exception:
            pass

    span_active_breakpoints.clear()
    active_core_stack_by_thread.clear()

    _delete_matching_existing_breakpoints()

    gdb.execute("set breakpoint pending on", to_string=True)
    gdb.execute("set pagination off", to_string=True)

    b1 = CoreBeginBreakpoint(TOKIO_CORE_POLL_BEGIN_SPEC, internal=False)
    b1.silent = True

    b2 = InstrumentedPollBreakpoint(TRACING_INSTRUMENT_POLL_SPEC, internal=False)
    b2.silent = True

    b3 = CoreResultBreakpoint(TOKIO_CORE_POLL_RESULT_SPEC, internal=False)
    b3.silent = True

    span_active_breakpoints.extend([b1, b2, b3])

    gdb.write("span_active_ring: installed breakpoints:\n")
    gdb.write(f"  Core::poll begin     {TOKIO_CORE_POLL_BEGIN_SPEC}\n")
    gdb.write(f"  Instrumented::poll   {TRACING_INSTRUMENT_POLL_SPEC} [always enabled, cheap inactive path]\n")
    gdb.write(f"  Core::poll result    {TOKIO_CORE_POLL_RESULT_SPEC} [records only Poll::Pending]\n")


class SpanActiveInstallCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-install", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        span_active_install()


class SpanActiveStatsCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-stats", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        active_total = sum(len(v) for v in active_core_stack_by_thread.values())

        gdb.write("\n=== span-active-stats ===\n")
        gdb.write(f"ring len/maxlen:       {len(span_active_ring)}/{span_active_ring.maxlen}\n")
        gdb.write(f"active threads:        {len(active_core_stack_by_thread)}\n")
        gdb.write(f"active stack entries:  {active_total}\n")
        gdb.write(f"tracked breakpoints:   {len(span_active_breakpoints)}\n")
        gdb.write(f"span seq:              {span_seq}\n")
        gdb.write(f"core pending seq:      {core_seq}\n")


class SpanActiveListCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-list", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = arg.split()
        n = int(args[0]) if args else 30
        entries = list(span_active_ring)[-n:]

        gdb.write(f"\n=== span-active-list: {len(entries)} of {len(span_active_ring)} ===\n")

        for e in entries:
            last = e["last_span"]
            where = f"{last['file']}:{last['line']}" if last else "<no Instrumented span seen>"
            gdb.write(
                f"seq={e['seq']} {e['thread']} "
                f"core={e['core_ptr']} last={where}\n"
            )


class SpanActiveDumpCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-dump", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = shlex.split(arg)
        n = int(args[0]) if len(args) >= 1 else 100
        path = os.path.expanduser(args[1]) if len(args) >= 2 else None

        entries = list(span_active_ring)[-n:]

        lines = []
        lines.append(f"=== span-active-dump: {len(entries)} of {len(span_active_ring)} ===")
        lines.append(f"RING_MAX={RING_MAX}")

        for e in entries:
            lines.append("")
            lines.append(f"--- pending Core::poll seq={e['seq']} ---")
            lines.append(f"thread:   {e['thread']}")
            lines.append(f"core_ptr: {e['core_ptr']}")

            last = e["last_span"]
            if last:
                lines.append(f"last_span: seq={last['seq']} {last['file']}:{last['line']}")
            else:
                lines.append("last_span: <none>")

        text = "\n".join(lines) + "\n"

        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
            gdb.write(f"span-active-dump wrote {len(text)} bytes to {path}\n")
        else:
            gdb.write(text)


class SpanActiveClearCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-clear", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        span_active_ring.clear()
        active_core_stack_by_thread.clear()
        gdb.write("span-active buffers cleared\n")


class SpanActiveDisableCommand(gdb.Command):
    def __init__(self):
        super().__init__("span-active-disable", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        for bp in span_active_breakpoints:
            try:
                if bp.is_valid():
                    bp.enabled = False
            except Exception:
                pass
        gdb.write("span-active breakpoints disabled\n")


def _safe_register(cls, name):
    try:
        cls()
    except Exception as ex:
        gdb.write(f"span_active_ring: command {name} not registered, probably already exists: {ex}\n")


_safe_register(SpanActiveInstallCommand, "span-active-install")
_safe_register(SpanActiveStatsCommand, "span-active-stats")
_safe_register(SpanActiveListCommand, "span-active-list")
_safe_register(SpanActiveDumpCommand, "span-active-dump")
_safe_register(SpanActiveClearCommand, "span-active-clear")
_safe_register(SpanActiveDisableCommand, "span-active-disable")


span_active_install()

gdb.write(
    "Loaded span_active_ring.py. Commands: "
    "span-active-stats, span-active-list, span-active-dump, "
    "span-active-clear, span-active-disable, span-active-install\n"
)