"""Offline checks for explicit turn savestate action codes."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bridge_client import (
    CHECKPOINTS_PER_LAP,
    NUM_ACTIONS,
    SAVE_ACTION,
    decode_bcd_progress,
    load_slot_action,
    save_slot_action,
)


def test_lap_timer_byte_is_decoded_from_packed_bcd():
    """The BCD decode was always right; what it decodes is the LAP TIMER.

    0x000C1 was read as track progress for roughly 40,000 training episodes.
    It is a clock: held still in mid-race it keeps counting while the integral
    of speed does not move. Progress now comes from the machine's world
    position instead (track_map.py), so CHECKPOINTS_PER_LAP is the resolution
    of that derived map rather than anything read out of RAM.
    """
    assert CHECKPOINTS_PER_LAP == 90
    assert [decode_bcd_progress(value) for value in (0x09, 0x10, 0x25, 0x59)] == [
        9, 10, 25, 59
    ]


def test_turn_slot_actions_never_overlap_driving_or_legacy_save():
    load_codes = {load_slot_action(slot) for slot in range(10)}
    save_codes = {save_slot_action(slot) for slot in range(10)}
    assert min(load_codes) > NUM_ACTIONS
    assert SAVE_ACTION not in load_codes | save_codes
    assert load_codes.isdisjoint(save_codes)


def test_turn_slot_actions_validate_bizhawk_slot_range():
    for invalid in (-1, 10):
        try:
            load_slot_action(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid slot {invalid} was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
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
