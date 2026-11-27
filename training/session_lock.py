"""Prevent two trainers from issuing actions to the SAME emulator instances.

Used to be a single, fully exclusive lock: whichever trainer got there first
blocked every other trainer until it finished, on ANY instance range at all.
That was safe but blunt -- it also blocked two trainers that were never
going to collide, e.g. DQN on instances 0-1 and the genetic search on 8-15,
which do not touch the same bridge files or shared-memory names at all.

Now the lock tracks WHICH instance numbers each holder occupies, as one file
per holder in a directory rather than one file for everyone. Acquiring
succeeds unless the requested range overlaps a range some other LIVE process
still holds; a holder whose process has died is cleaned up automatically.
Nothing here polices whether an instance range is actually correct, or
whether two on the SAME range would still crash into each other -- it only
prevents the specific case this project has already been burned by.

    with EmulatorSessionLock("dqn training", offset=0, count=8):
        ...

Held files live in runs/locks/<pid>.json, e.g.:
    {"pid": 1234, "label": "dqn training", "offset": 0, "count": 8,
     "started": 1787..., "command": "training\\train_dqn.py --instances 8"}
"""

import ctypes
import glob
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, "runs", "locks")

# Where the OLD single-file lock used to live. Cleared on import so a stale
# copy from before this rewrite can never look like a live holder.
_LEGACY_LOCK_PATH = os.path.join(BASE_DIR, "runs", "emulator_training.lock")


def process_is_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def ranges_overlap(start_a, count_a, start_b, count_b):
    return start_a < start_b + count_b and start_b < start_a + count_a


def _clear_legacy_lock():
    try:
        os.unlink(_LEGACY_LOCK_PATH)
    except FileNotFoundError:
        pass
    except OSError:
        pass


_clear_legacy_lock()


class EmulatorSessionLock:
    """Held for as long as this process is driving instances [offset, offset+count)."""

    def __init__(self, label, offset=0, count=None, directory=LOCK_DIR):
        self.label = str(label)
        self.offset = int(offset)
        # None means "the whole space" -- the old fully-exclusive behaviour,
        # for anything that genuinely cannot say which instances it will use.
        self.count = None if count is None else int(count)
        self.directory = directory
        self.path = os.path.join(directory, f"{os.getpid()}.json")
        self.acquired = False

    def _my_range_covers(self, offset, count):
        if self.count is None:
            return True
        return not ranges_overlap(self.offset, self.count, offset, count)

    def _live_holders(self):
        """Other holders' records, dropping and deleting any that are dead."""
        holders = []
        try:
            paths = glob.glob(os.path.join(self.directory, "*.json"))
        except OSError:
            return holders

        for path in paths:
            if os.path.basename(path) == f"{os.getpid()}.json":
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, ValueError, TypeError):
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue

            if process_is_alive(record.get("pid")):
                holders.append(record)
            else:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return holders

    def acquire(self):
        os.makedirs(self.directory, exist_ok=True)

        for holder in self._live_holders():
            holder_offset = holder.get("offset", 0)
            holder_count = holder.get("count")
            # A holder with no recorded count is the old-style "everything"
            # lock, or this process itself asking for the whole space --
            # either way, nothing can run alongside it.
            overlaps = (
                holder_count is None
                or self.count is None
                or ranges_overlap(self.offset, self.count, holder_offset, holder_count)
            )
            if overlaps:
                raise SystemExit(
                    "Another emulator trainer already owns overlapping "
                    f"instances: {holder.get('label', 'unknown')} "
                    f"(PID {holder.get('pid')}, instances "
                    f"{holder_offset}..{holder_offset + (holder_count or 0) - 1}"
                    f"{'+' if holder_count is None else ''}).\n"
                    "Stop that run, or use --offset to pick a disjoint range."
                )

        record = {
            "pid": os.getpid(),
            "label": self.label,
            "offset": self.offset,
            "count": self.count,
            "started": time.time(),
            "command": " ".join(sys.argv),
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

        self.acquired = True
        return self

    def release(self):
        if not self.acquired:
            return
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self.acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


if __name__ == "__main__":
    # Batch-file preflight: acquire/release once, for the whole space, which
    # also clears any of THIS process's own stale entries. Run before the
    # launcher stops or starts EmuHawk.
    with EmulatorSessionLock("launcher preflight"):
        pass
