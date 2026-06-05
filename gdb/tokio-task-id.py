python
import gdb

def _as_u64(v):
    try:
        return int(v)
    except Exception:
        pass

    for t in ("unsigned long long", "unsigned long", "uintptr_t"):
        try:
            return int(v.cast(gdb.lookup_type(t)))
        except Exception:
            pass

    raise ValueError("cannot convert value to integer")


def _read_u64(addr):
    mem = gdb.selected_inferior().read_memory(addr, 8)
    return int.from_bytes(bytes(mem), "little")


def _eval_in_frame(frame, expr):
    old = gdb.selected_frame()
    try:
        frame.select()
        return gdb.parse_and_eval(expr)
    finally:
        old.select()


def _try_pointer(frame, ptr_expr, route):
    """
    ptr_expr should evaluate either to:
      - *mut Header
      - *mut Cell<T, S>

    Header is the first field of Cell, so the address base is the same.
    """
    p = _eval_in_frame(frame, ptr_expr)
    base = _as_u64(p)

    obj = p.dereference()

    # raw::poll route: p is *mut Header
    try:
        off = _as_u64(obj["vtable"]["id_offset"])
        task_id = _read_u64(base + off)
        return task_id, route + " Header"
    except Exception:
        pass

    # Harness route: p is *mut Cell<T, S>
    try:
        off = _as_u64(obj["header"]["vtable"]["id_offset"])
        task_id = _read_u64(base + off)
        return task_id, route + " Cell"
    except Exception:
        pass

    return None


def tokio_task_id():
    old = gdb.selected_frame()

    try:
        thread = gdb.parse_and_eval("$_thread")
    except Exception:
        thread = "?"

    try:
        pc = gdb.parse_and_eval("$pc")
    except Exception:
        pc = "?"

    frame = gdb.newest_frame()
    idx = 0

    while frame is not None:
        candidates = [
            # raw::poll(ptr=...)
            ("ptr.pointer", "raw::poll ptr.pointer"),
            ("ptr.pointer.pointer", "raw::poll ptr.pointer.pointer"),
            ("ptr", "raw::poll ptr"),

            # Harness::poll(self=...)
            ("self.cell.pointer", "Harness self.cell.pointer"),
            ("self.cell.pointer.pointer", "Harness self.cell.pointer.pointer"),
            ("(*self).cell.pointer", "Harness (*self).cell.pointer"),
            ("(*self).cell.pointer.pointer", "Harness (*self).cell.pointer.pointer"),
        ]

        for expr, route in candidates:
            try:
                result = _try_pointer(frame, expr, route)
                if result is not None:
                    task_id, used_route = result
                    fn = frame.name() or "?"
                    print(f"XXX task_id={task_id} thread={thread} pc={pc} frame=#{idx} route={used_route}")
                    print(f"    frame_fn={fn}")
                    old.select()
                    return
            except Exception:
                pass

        frame = frame.older()
        idx += 1

    print(f"XXX task_id=<not found> thread={thread} pc={pc}")
    old.select()


class TokioTaskIdCommand(gdb.Command):
    def __init__(self):
        super(TokioTaskIdCommand, self).__init__("tokio_task_id", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        tokio_task_id()


TokioTaskIdCommand()
end