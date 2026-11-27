"""Fast tests for the permanent geometry-based driving guard rails."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bridge_client import (
    SHOULDER_L,
    SHOULDER_NONE,
    SHOULDER_R,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
)
from env.driving_assist import DrivingAssist, centre_signal, curve_ahead_signal


def reading(left, right, on_road=True, nearest=None):
    return {
        "left_px": left,
        "right_px": right,
        "on_road": on_road,
        "nearest_road_x": nearest,
        "center_x": 128,
    }


def rows(left, right):
    return [reading(left, right) for _ in range(4)]


def test_centre_signal_points_away_from_nearer_wall():
    assert centre_signal(rows(10, 70)) > 0, "left wall close: move right"
    assert centre_signal(rows(70, 10)) < 0, "right wall close: move left"
    assert centre_signal(rows(50, 50)) == 0.0


def test_warning_overrides_toward_safety_without_opposite_lean():
    assist = DrivingAssist()
    assist.observe(rows(20, 70))
    action, info = assist.apply(STEER_STRAIGHT, SHOULDER_L)
    assert action == (STEER_RIGHT, SHOULDER_NONE)
    assert info["override"] and info["reason"] == "predicted_wall"


def test_emergency_uses_matching_lean_but_lean_remains_policy_selectable():
    assist = DrivingAssist()
    assist.observe(rows(0, 70))
    action, info = assist.apply(STEER_LEFT, SHOULDER_L)
    assert action == (STEER_RIGHT, SHOULDER_R)
    assert info["reason"] == "emergency"

    assist.reset()
    assist.observe(rows(70, 70))
    action, info = assist.apply(STEER_LEFT, SHOULDER_L)
    assert action == (STEER_LEFT, SHOULDER_L), "safe lean-left must reach the game"
    assert not info["override"]


def test_closing_wall_triggers_before_contact():
    assist = DrivingAssist(prediction_frames=6)
    assist.observe(rows(60, 80))
    state = assist.observe(rows(42, 80))
    assert state["predicted_left"] < 42
    assert state["risk"] > 0
    action, _ = assist.apply(STEER_STRAIGHT, SHOULDER_NONE)
    assert action[0] == STEER_RIGHT


def test_rapid_reversal_is_debounced_but_emergency_is_not():
    assist = DrivingAssist(reversal_hold=6)
    assist.observe(rows(70, 70))
    assert assist.apply(STEER_LEFT, SHOULDER_L)[0] == (STEER_LEFT, SHOULDER_L)
    action, info = assist.apply(STEER_RIGHT, SHOULDER_R)
    assert action == (STEER_STRAIGHT, SHOULDER_NONE)
    assert info["reversal_blocked"]

    assist.observe(rows(0, 70))
    action, info = assist.apply(STEER_RIGHT, SHOULDER_R)
    assert action == (STEER_RIGHT, SHOULDER_R)
    assert not info["reversal_blocked"]


def test_confirmed_curve_ahead_steers_and_leans_before_near_wall_risk():
    assist = DrivingAssist()
    curve = [
        reading(15, 75),
        reading(20, 70),
        reading(60, 60),
        reading(60, 60),
    ]
    state = assist.observe(curve)
    assert curve_ahead_signal(curve) > 0.22
    assert state["risk"] == 0.0, "near row is still safe"
    action, info = assist.apply(STEER_STRAIGHT, SHOULDER_NONE)
    assert action == (STEER_RIGHT, SHOULDER_R)
    assert info["reason"] == "lookahead_turn"
    # An ordinary curve shortens the hold but does NOT collapse it to a
    # single frame. Sharing turn_lean_threshold with action_hold used to do
    # exactly that, and |curve_signal| has a median of 0.500 on real driving,
    # so 91.9% of decisions lasted one frame and the steering flickered
    # instead of being held.
    assert assist.action_hold(4) == 3


def test_emergency_only_mode_leaves_warnings_and_curves_to_the_policy():
    assist = DrivingAssist(mode="emergency")
    curve = [reading(15, 75), reading(20, 70), reading(60, 60), reading(60, 60)]
    assist.observe(curve)
    requested = (STEER_LEFT, SHOULDER_L)
    action, info = assist.apply(*requested)
    assert action == requested
    assert not info["override"]

    assist.observe(rows(0, 70))
    action, info = assist.apply(STEER_LEFT, SHOULDER_L)
    assert action == (STEER_RIGHT, SHOULDER_R)
    assert info["override"] and info["reason"] == "emergency"


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


def test_only_an_emergency_collapses_the_action_hold_to_one_frame():
    """One frame is the emergency escape hatch, not the normal case.

    Measured over 47180 frames of a live run, the previous thresholds fired on
    91.9% of decisions, so the policy re-chose at 60Hz and changed direction on
    45% of frames. Steering authority is not the problem -- a held input pulls
    the machine 35 world units off line in 5 frames and 412 in 59 -- but the
    input has to survive more than one frame to do it.
    """
    assist = DrivingAssist()

    # Off the road entirely: react now.
    assist.observe(rows(0, 70))
    assert assist.state["emergency"]
    assert assist.action_hold(4) == 1

    # Clear road, no curve: the policy keeps the whole frame skip.
    assist.observe(rows(60, 60))
    assert not assist.state["emergency"]
    assert assist.action_hold(4) == 4


def test_action_hold_never_exceeds_the_frame_skip_it_is_given():
    assist = DrivingAssist()
    assist.observe(rows(60, 60))
    for skip in (1, 2, 3, 4, 8):
        assert 1 <= assist.action_hold(skip) <= skip
