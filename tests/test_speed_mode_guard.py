"""A policy must never be run under a host speed it was not trained at.

Above 50% the emulator stops being reproducible: replaying identical inputs
from one savestate, 30 emulated frames of left against right, the world-space
separation of the endpoints over 10 repeats is 75.18 +/- 0.35 at 50% and
101.41 +/- 11.39 at 100%, where every single repeat differed and the range ran
74.24 to 116.07. That spread straddles the 50% value, which is why one
3-repeat measurement called 100% "11% weaker" and a 10-repeat one called it
"32% stronger" -- neither direction is a real property, but the loss of
determinism is.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bridge_client
from bridge_client import ALLOW_SPEED_MODE_MISMATCH, REQUIRED_SPEED_MODE


class Probe(bridge_client.BridgeClient):
    """A client with no emulator behind it, to exercise the guard alone."""

    def __init__(self, reported):
        self._speed_mode = reported


def without_override(function):
    previous = os.environ.pop(ALLOW_SPEED_MODE_MISMATCH, None)
    try:
        return function()
    finally:
        if previous is not None:
            os.environ[ALLOW_SPEED_MODE_MISMATCH] = previous


def test_matching_speed_mode_is_accepted():
    without_override(lambda: Probe(REQUIRED_SPEED_MODE).check_speed_mode())


def test_a_faster_host_is_refused_rather_than_silently_driven():
    def attempt():
        try:
            Probe(100).check_speed_mode()
        except RuntimeError as exc:
            assert "100" in str(exc) and str(REQUIRED_SPEED_MODE) in str(exc)
            return True
        return False

    assert without_override(attempt), (
        "running at 100% must fail loudly; a silent mismatch changes the "
        "physics out from under a trained policy"
    )


def test_a_slower_host_is_also_refused():
    """Slower is still different. 25% is deterministic but not the same."""
    def attempt():
        try:
            Probe(25).check_speed_mode()
        except RuntimeError:
            return True
        return False

    assert without_override(attempt)


def test_a_bridge_that_does_not_report_its_speed_is_not_blocked():
    """An older bridge.lua cannot be checked, so it is not falsely failed."""
    without_override(lambda: Probe(None).check_speed_mode())


def test_the_override_exists_for_tools_that_measure_the_difference():
    os.environ[ALLOW_SPEED_MODE_MISMATCH] = "1"
    try:
        Probe(100).check_speed_mode()
    finally:
        os.environ.pop(ALLOW_SPEED_MODE_MISMATCH, None)


def test_training_launcher_requests_exactly_the_required_speed():
    from tools.launch_instances import TRAINING_SPEED_MODE, training_environment

    assert TRAINING_SPEED_MODE == REQUIRED_SPEED_MODE, (
        "the launcher and the guard must not be able to drift apart"
    )
    assert training_environment(3)["FZERO_SPEEDMODE"] == str(REQUIRED_SPEED_MODE)


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(bool(failures))
