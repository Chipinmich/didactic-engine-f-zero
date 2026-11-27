"""Tests for the steering controller.

The controller has a lot of tunable thresholds and no way to check them
against the game without driving, so these pin down the behaviour that must
hold regardless of tuning.

    python tests/test_control.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (
    SHOULDER_NONE,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    THROTTLE_COAST,
    TOP_SPEED_RAW,
)
from control import Controller

HALF_WIDTH = 128.0

# Measured peak raw speed is about 2034, so "fast" must be under that --
# anything above the top speed is correctly read as over target and coasted.
FAST = 1950
SLOW = 100

THROTTLE_NAMES = {THROTTLE_ACCEL: "ACCEL", THROTTLE_COAST: "COAST"}


def rows(*pairs):
    """Readings from (left_px, right_px) pairs, near row first."""
    return [
        {
            "left_px": left,
            "right_px": right,
            "scan_y": 0,
            "center_x": 128,
            "on_road": True,
            "nearest_road_x": None,
        }
        for (left, right) in pairs
    ]


def off_road_rows(nearest_road_x):
    """Readings for a car that has left the track entirely."""
    return [
        {
            "left_px": 0,
            "right_px": 0,
            "scan_y": 0,
            "center_x": 128,
            "on_road": False,
            "nearest_road_x": nearest_road_x,
        }
        for _ in range(3)
    ]


def steer_counts(readings, frames=60, speed=FAST):
    controller = Controller()
    out = [controller.update(readings, HALF_WIDTH, speed)[0][0] for _ in range(frames)]
    return (
        sum(1 for s in out if s == STEER_LEFT),
        sum(1 for s in out if s == STEER_RIGHT),
        sum(1 for s in out if s == STEER_STRAIGHT),
    )


def test_centred_runs_straight():
    left, right, _ = steer_counts(rows((60, 60), (60, 60), (60, 60)))
    assert left == 0 and right == 0, "a centred car should not steer"


def test_steers_away_from_walls():
    _, right, _ = steer_counts(rows((5, 120), (10, 120), (20, 120)))
    assert right > 0, "close to the left wall should steer right"

    left, _, _ = steer_counts(rows((120, 5), (120, 10), (120, 20)))
    assert left > 0, "close to the right wall should steer left"


def test_steering_is_proportional():
    """Further off centre means steering on more frames, not just harder."""
    fractions = []
    for left, right in [(58, 66), (40, 90), (10, 120)]:
        l, r, s = steer_counts(rows((left, right), (left, right), (left, right)), frames=200)
        fractions.append((l + r) / 200.0)

    assert fractions[0] < fractions[1] < fractions[2], (
        f"steering rate should rise with error, got {fractions}"
    )


def test_lean_only_for_large_corrections():
    controller = Controller()
    (_, _, shoulder), _ = controller.update(
        rows((60, 65), (60, 65), (60, 65)), HALF_WIDTH, FAST
    )
    assert shoulder == SHOULDER_NONE, "a centred car should not lean"

    # Lean follows the smoothed demand, so it takes a few frames to build.
    # That is deliberate: leaning on a single noisy frame chatters the
    # shoulder button.
    controller = Controller()
    hard_left_wall = rows((5, 120), (10, 120), (20, 120))
    shoulder = SHOULDER_NONE
    for _ in range(20):
        (_, _, shoulder), _ = controller.update(hard_left_wall, HALF_WIDTH, FAST)
    assert shoulder != SHOULDER_NONE, "a sustained large correction should lean"


def test_throttle_responds_to_curvature_not_position():
    """Off-centre on a straight must not be mistaken for a corner."""
    controller = Controller()
    (_, throttle, _), _ = controller.update(
        rows((60, 60), (60, 60), (60, 60)), HALF_WIDTH, FAST
    )
    assert throttle == THROTTLE_ACCEL, "a straight should stay on the throttle"

    controller = Controller()
    (_, throttle, _), _ = controller.update(
        rows((8, 118), (10, 116), (12, 114)), HALF_WIDTH, FAST
    )
    assert throttle == THROTTLE_ACCEL, (
        "hugging a wall down a straight is not a corner -- must not lift"
    )

    controller = Controller()
    (_, throttle, _), _ = controller.update(
        rows((60, 60), (5, 125), (2, 126)), HALF_WIDTH, FAST
    )
    assert throttle == THROTTLE_COAST, "a sharp corner ahead at speed should lift"


def test_open_road_ahead_is_not_a_corner():
    """A wall beside the car with open road ahead must not read as a corner.

    A row where the road runs off both sides carries no curvature
    information. Treating it as "centred" makes the difference against a
    near row pinned to a wall look like maximum curvature, which had the
    car braking almost continuously and crawling round the track.
    """
    controller = Controller()
    (_, throttle, _), reason = controller.update(
        rows((0, None), (None, None), (None, None)), HALF_WIDTH, FAST
    )
    assert throttle == THROTTLE_ACCEL, (
        f"scraping a wall with open road ahead should not brake ({reason})"
    )


def test_never_lifts_at_low_speed():
    controller = Controller()
    (_, throttle, _), _ = controller.update(
        rows((60, 60), (5, 125), (2, 126)), HALF_WIDTH, SLOW
    )
    assert throttle == THROTTLE_ACCEL, "lifting while already slow only loses time"


def test_target_speed_falls_with_curvature():
    """A sharper corner ahead must lower the target, monotonically."""
    controller = Controller()
    targets = [controller.target_speed(s) for s in (0.0, 0.2, 0.4, 0.8)]
    assert targets[0] == TOP_SPEED_RAW, "a straight should target top speed"
    assert targets[0] >= targets[1] >= targets[2] >= targets[3], (
        f"target should fall as curvature rises, got {targets}"
    )
    assert targets[-1] >= controller.min_corner_speed


def test_gets_back_on_throttle_under_target():
    """Once slowed to the target, the throttle must come straight back on.

    Coasting past the target is what loses the time: it sheds 80 a frame and
    the throttle only wins back 19.
    """
    controller = Controller()
    corner = rows((60, 60), (5, 125), (2, 126))

    target = None
    for speed in (2100, 1800, 1500, 1200, 900):
        (_, throttle, _), _ = controller.update(corner, HALF_WIDTH, speed)
        target = controller.last_target
        if speed <= target:
            assert throttle == THROTTLE_ACCEL, (
                f"at {speed} (target {target:.0f}) should be back on throttle"
            )


def test_does_not_coast_for_a_trivial_overshoot():
    """A single coast frame sheds 80, so do not coast to shave less."""
    controller = Controller()
    corner = rows((60, 60), (5, 125), (2, 126))
    controller.update(corner, HALF_WIDTH, FAST)
    target = controller.last_target

    (_, throttle, _), reason = controller.update(
        corner, HALF_WIDTH, int(target + 20)
    )
    assert throttle == THROTTLE_ACCEL, (
        f"20 over target is less than one coast frame sheds ({reason})"
    )


def test_noisy_signal_does_not_oscillate():
    """A car near the centre with noisy readings must not saw at the wheel.

    The road is dithered and there are only three scan rows, so the raw
    per-frame demand jitters around zero. Steering on it unfiltered measured
    2.1 direction reversals a second on a real run, which scrubs speed and
    never settles the car.
    """
    import random

    random.seed(1)
    controller = Controller()

    directions = []
    for _ in range(600):
        # Jitter either side of centred, no real corner. The range has to
        # cross the deadzone, or the deadzone alone absorbs it and the test
        # proves nothing -- unfiltered, this measures 2.7 reversals/s,
        # matching what a real run showed.
        left = max(60 + random.randint(-25, 25), 0)
        right = max(60 + random.randint(-25, 25), 0)
        (steer, _, _), _ = controller.update(
            rows((left, right), (left, right), (left, right)), HALF_WIDTH, FAST
        )
        if steer != STEER_STRAIGHT:
            directions.append(steer)

    reversals = sum(
        1 for a, b in zip(directions, directions[1:]) if a != b
    )
    # 600 frames is 10 seconds at 60fps.
    per_second = reversals / 10.0
    assert per_second < 1.0, (
        f"steering oscillates: {per_second:.1f} reversals/s over 10s"
    )


def test_off_road_steers_back_toward_the_road():
    """Leaving the track must not read as being perfectly centred.

    Off the road both edges measure zero, which is indistinguishable from a
    centred car unless on_road is checked. The controller then held
    straight at full throttle against a wall indefinitely -- two runs spent
    59% of their frames in exactly that state.
    """
    left, right, _ = steer_counts(off_road_rows(200), frames=30, speed=500)
    assert right > 25 and left == 0, (
        f"road to the right should steer right, got left={left} right={right}"
    )

    left, right, _ = steer_counts(off_road_rows(40), frames=30, speed=500)
    assert left > 25 and right == 0, (
        f"road to the left should steer left, got left={left} right={right}"
    )


def test_recovery_direction_is_latched():
    """Off the road, the way back must be committed to, not re-decided.

    Measured off the road, the reported side of the road flips on 18% of
    frames. Recomputing the direction each frame lets the smoothing average
    those contradictions to nearly nothing, so the machine coasts off-track
    until it stops -- and a stationary machine cannot steer back.
    """
    controller = Controller()

    # Establish the recovery direction with the road to the right.
    controller.update(off_road_rows(220), HALF_WIDTH, 500)

    # Now the reading flips side, as it really does. The committed
    # direction must hold.
    steers = []
    for step in range(20):
        noisy = off_road_rows(40 if step % 2 else 220)
        steers.append(controller.update(noisy, HALF_WIDTH, 500)[0][0])

    assert all(s == STEER_RIGHT for s in steers), (
        f"recovery should hold one direction through noise, got {steers}"
    )


def test_recovery_releases_once_back_on_road():
    """The latch must not survive getting back on the track."""
    controller = Controller()
    controller.update(off_road_rows(220), HALF_WIDTH, 500)
    assert controller._recovery_direction is not None

    controller.update(rows((60, 60), (60, 60), (60, 60)), HALF_WIDTH, FAST)
    assert controller._recovery_direction is None, "latch should clear on the road"


def test_recovery_stays_on_the_throttle():
    """Steering needs speed, so recovery must not lift off."""
    from bridge_client import THROTTLE_ACCEL as ACCEL

    controller = Controller()
    (_, throttle, _), _ = controller.update(off_road_rows(220), HALF_WIDTH, 500)
    assert throttle == ACCEL, "coasting off-road stops the machine steering back"


def test_off_road_does_not_skew_curvature():
    """An off-road row says nothing about where the track goes."""
    controller = Controller()
    (_, throttle, _), _ = controller.update(off_road_rows(200), HALF_WIDTH, 500)
    assert controller.last_severity == 0.0, (
        f"off-road rows must not register as curvature, got {controller.last_severity}"
    )


def test_blind_frame_holds_last_correction():
    """No readings at all (undecodable frame) must not snap the wheel straight."""
    controller = Controller()
    for _ in range(10):
        controller.update(rows((5, 120), (10, 120), (20, 120)), HALF_WIDTH, FAST)

    out = [controller.update([], HALF_WIDTH, FAST)[0][0] for _ in range(30)]
    assert any(s == STEER_RIGHT for s in out), (
        "should hold the correction when there is no reading at all"
    )


def test_wide_open_road_runs_straight():
    """Both walls out of view means the road fills the screen.

    There is no centre to aim at, so this must run straight. Treating the
    open sides as walls at half the frame width invents an offset and makes
    the car flap between full-lock left and right.
    """
    wide = rows((None, None), (None, None), (None, None))
    left, right, _ = steer_counts(wide, frames=60)
    assert left == 0 and right == 0, (
        f"wide-open road should not steer, got left={left} right={right}"
    )


def test_one_visible_wall_only_reacts_when_close():
    """Road running off-screen one way is not a reason to steer."""
    # Wall on the left but comfortably far, road open right: no correction.
    left, right, _ = steer_counts(
        rows((90, None), (90, None), (90, None)), frames=60
    )
    assert left == 0 and right == 0, "a distant wall should not provoke steering"

    # Same geometry but the wall is close: steer away from it.
    left, right, _ = steer_counts(
        rows((8, None), (8, None), (8, None)), frames=60
    )
    assert right > 0 and left == 0, "a close left wall should steer right"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")

    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
