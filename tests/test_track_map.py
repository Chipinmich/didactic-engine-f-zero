"""Progress must measure where the car is, not how long it has been racing."""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from track_map import LOOK_BACK, TrackMap


def circle_map(count=90, radius=1000.0):
    """A synthetic closed loop, so these tests need no emulator or capture."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    track = TrackMap.__new__(TrackMap)
    track.centreline = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], 1
    ).astype(np.float64)
    track.checkpoints = count
    return track


def test_a_stationary_car_makes_no_progress():
    """The bug this whole mechanism exists to prevent.

    The old reading (RAM 0x000C1) gained 9 checkpoints over 124 frames with the
    speed reading at exactly 0 and the distance integral frozen, because it was
    the lap timer. Position cannot do that.
    """
    track = circle_map()
    x, y = track.centreline[20]
    index = track.nearest(x, y)
    for _ in range(200):
        index, _ = track.advance(x, y, index)
    assert index == 20


def test_progress_follows_the_car_round_the_track():
    track = circle_map()
    index = None
    seen = []
    for point in track.centreline:
        index, _ = track.advance(point[0], point[1], index)
        seen.append(index)
    assert seen == list(range(track.checkpoints))


def test_crossing_the_line_does_not_read_as_driving_backwards():
    """Checkpoint 89 and checkpoint 0 are adjacent in space.

    A global nearest-point lookup handles this fine on its own, but the caller
    turns consecutive readings into a gain -- so the value must wrap forwards,
    and advance() must report the wrap rather than silently resetting.
    """
    track = circle_map()
    index = track.checkpoints - 1
    following = track.centreline[0]
    index, wrapped = track.advance(following[0], following[1], index)
    assert index == 0
    assert wrapped, "the caller cannot tell a lap from a 89-checkpoint reversal"


def test_the_search_window_cannot_jump_to_the_far_side_of_the_track():
    """Real tracks pass close to themselves; a windowed search is why that is safe.

    Placed exactly on the opposite side of the loop, a global lookup would jump
    there. The window keeps the answer near where the car actually was.
    """
    track = circle_map()
    opposite = track.centreline[45]
    index, _ = track.advance(opposite[0], opposite[1], 0)
    assert index <= 12 or index >= track.checkpoints - LOOK_BACK


def test_a_short_spin_backwards_is_allowed_but_a_lap_is_not():
    track = circle_map()
    behind = track.centreline[8]
    index, _ = track.advance(behind[0], behind[1], 10)
    assert index == 8


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
