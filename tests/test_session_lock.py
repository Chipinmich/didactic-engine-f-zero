"""Tests for the range-aware emulator session lock.

No emulator needed -- this only exercises the lock's own file-based logic.
The one property that matters most: two trainers on DISJOINT instance
ranges must both be able to hold the lock at once, since that is the whole
point of this rewrite (DQN on 0-7 alongside the genetic search on 8-15).
"""

import glob
import multiprocessing
import os
import shutil
import sys
import tempfile
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from training.session_lock import EmulatorSessionLock, ranges_overlap


def test_ranges_overlap():
    assert ranges_overlap(0, 8, 4, 8)   # 0-7 and 4-11 share 4-7
    assert ranges_overlap(0, 8, 7, 1)   # 0-7 and just 7
    assert not ranges_overlap(0, 8, 8, 8)   # 0-7 and 8-15, adjacent, no overlap
    assert not ranges_overlap(8, 8, 0, 8)   # same, reversed argument order


def _acquire_and_hold(directory, label, offset, count, hold_seconds, outcome_path):
    """Run in a SEPARATE process, so process_is_alive sees a real PID."""
    lock = EmulatorSessionLock(label, offset=offset, count=count, directory=directory)
    try:
        lock.acquire()
        with open(outcome_path, "w") as fh:
            fh.write("acquired")
        time.sleep(hold_seconds)
    except SystemExit as exc:
        with open(outcome_path, "w") as fh:
            fh.write(f"blocked:{exc}")
    finally:
        lock.release()


def _run_holder(directory, label, offset, count, hold_seconds):
    outcome_path = os.path.join(directory, f"outcome_{label}.txt")
    process = multiprocessing.Process(
        target=_acquire_and_hold,
        args=(directory, label, offset, count, hold_seconds, outcome_path),
    )
    process.start()
    return process, outcome_path


def _wait_for_outcome(path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path) as fh:
                content = fh.read()
            if content:
                return content
        time.sleep(0.05)
    raise TimeoutError(f"no outcome written to {path}")


def test_disjoint_ranges_can_both_acquire():
    """The whole point: DQN on 0-7 and the genetic search on 8-15 at once."""
    directory = tempfile.mkdtemp()
    try:
        first, first_outcome = _run_holder(directory, "dqn", 0, 8, hold_seconds=1.5)
        # Give the first process time to actually write its lock file before
        # the second tries to acquire -- otherwise this is a race, not a test.
        time.sleep(0.5)
        second, second_outcome = _run_holder(directory, "genetic", 8, 8, hold_seconds=0.1)

        assert _wait_for_outcome(first_outcome) == "acquired"
        assert _wait_for_outcome(second_outcome) == "acquired", (
            "a disjoint range must not be blocked by another live holder"
        )

        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_overlapping_ranges_block():
    directory = tempfile.mkdtemp()
    try:
        first, first_outcome = _run_holder(directory, "dqn", 0, 8, hold_seconds=1.5)
        time.sleep(0.5)
        second, second_outcome = _run_holder(directory, "other_dqn", 4, 8, hold_seconds=0.1)

        assert _wait_for_outcome(first_outcome) == "acquired"
        outcome = _wait_for_outcome(second_outcome)
        assert outcome.startswith("blocked:"), (
            f"overlapping ranges (0-7 and 4-11) must block, got {outcome!r}"
        )

        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _acquire_release_immediately(directory, label, offset, count, outcome_path):
    lock = EmulatorSessionLock(label, offset=offset, count=count, directory=directory)
    try:
        lock.acquire()
        with open(outcome_path, "w") as fh:
            fh.write(f"acquired:{lock.path}")
        time.sleep(0.4)
    except SystemExit as exc:
        with open(outcome_path, "w") as fh:
            fh.write(f"blocked:{exc}")
    finally:
        lock.release()


def test_none_count_is_fully_exclusive_both_ways():
    """A holder with no known range (legacy/whole-space) blocks everything,
    and is itself blocked by anything else -- preserves the old behaviour
    for a caller that genuinely cannot say which instances it will touch.

    Two SEPARATE processes, like every other multi-holder test here: two
    EmulatorSessionLock objects created in-process would share os.getpid()
    and therefore the same lock filename, which is not how two real
    trainers (each its own OS process) ever collide.
    """
    directory = tempfile.mkdtemp()
    try:
        whole, whole_outcome = _run_holder(directory, "whole", 0, None, hold_seconds=1.2)
        time.sleep(0.5)

        narrow_outcome = os.path.join(directory, "outcome_narrow.txt")
        narrow_process = multiprocessing.Process(
            target=_acquire_release_immediately,
            args=(directory, "narrow", 100, 1, narrow_outcome),
        )
        narrow_process.start()

        assert _wait_for_outcome(whole_outcome) == "acquired"
        outcome = _wait_for_outcome(narrow_outcome)
        assert outcome.startswith("blocked:"), (
            f"a whole-space holder must block even a far-away range, got {outcome!r}"
        )

        whole.join(timeout=5)
        narrow_process.join(timeout=5)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_a_dead_holders_lock_is_cleaned_up():
    """A stale lock file from a PID that no longer exists must not block."""
    directory = tempfile.mkdtemp()
    try:
        # A PID essentially guaranteed not to be alive.
        fake_pid = 999999
        stale = EmulatorSessionLock("ghost", offset=0, count=8, directory=directory)
        stale.path = os.path.join(directory, f"{fake_pid}.json")
        import json
        with open(stale.path, "w") as fh:
            json.dump(
                {"pid": fake_pid, "label": "ghost", "offset": 0, "count": 8,
                 "started": time.time(), "command": "nothing"},
                fh,
            )

        lock = EmulatorSessionLock("live", offset=0, count=8, directory=directory)
        lock.acquire()  # must not raise
        assert not os.path.exists(stale.path), "the dead holder's file should be swept"
        lock.release()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_release_only_removes_its_own_file():
    """Two separate processes: same reasoning as the test above -- two
    EmulatorSessionLock objects sharing os.getpid() would share a filename,
    which is not how two real trainer processes ever collide."""
    directory = tempfile.mkdtemp()
    try:
        a, a_outcome = _run_holder(directory, "a", 0, 4, hold_seconds=1.2)
        time.sleep(0.4)

        b_outcome = os.path.join(directory, "outcome_b.txt")
        b_process = multiprocessing.Process(
            target=_acquire_release_immediately,
            args=(directory, "b", 4, 4, b_outcome),
        )
        b_process.start()

        assert _wait_for_outcome(a_outcome) == "acquired"
        outcome = _wait_for_outcome(b_outcome)
        assert outcome.startswith("acquired:"), f"disjoint range must acquire, got {outcome!r}"
        b_path = outcome.split(":", 1)[1]

        # b released and exited already (0.4s hold, well under a's 1.2s), but
        # a is still holding -- a's file must still be there, and only a's.
        b_process.join(timeout=5)
        assert not os.path.exists(b_path), "b should have released its own file"
        remaining = glob.glob(os.path.join(directory, "*.json"))
        assert len(remaining) == 1, (
            f"only a's lock file should remain while a still holds, found {remaining}"
        )

        a.join(timeout=5)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def run():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
