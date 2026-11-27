"""Offline safety checks for the distributed checkpoint-state builder."""

import os
import tempfile

from tools.make_turn_states import load_saved_slots, save_slots


def test_checkpoint_slots_start_with_only_measured_safe_slots():
    path = os.path.join(tempfile.gettempdir(), "missing_fzero_checkpoint_slots.txt")
    assert load_saved_slots(path, 2) == (2, 4)


def test_checkpoint_slots_are_persisted_atomically_for_the_next_ga_run():
    descriptor, path = tempfile.mkstemp(prefix="fzero_checkpoint_slots_")
    os.close(descriptor)
    try:
        save_slots(path, (2, 4, 5, 6))
        assert load_saved_slots(path, 2) == (2, 4, 5, 6)
    finally:
        os.unlink(path)
