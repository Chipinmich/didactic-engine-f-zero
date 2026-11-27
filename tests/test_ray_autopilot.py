"""Offline behavior checks for the standalone seven-ray autopilot."""

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
    THROTTLE_ACCEL,
)
from ray_autopilot import AutopilotMetrics, SevenRayAutopilot


def rows(left=60, right=60, on_road=True, nearest=None):
    return [
        {
            "left_px": left,
            "right_px": right,
            "on_road": on_road,
            "nearest_road_x": nearest,
            "center_x": 128,
        }
        for _ in range(4)
    ]


def actions(controller, readings, rays, frames=80):
    return [controller.update(readings, rays, 1500)[0] for _ in range(frames)]


def test_open_centred_road_accelerates_straight_without_lean():
    output = actions(SevenRayAutopilot(), rows(), [1.0] * 7, 40)
    assert all(action == (STEER_STRAIGHT, THROTTLE_ACCEL, SHOULDER_NONE)
               for action in output)


def test_left_obstacle_turns_and_leans_right():
    rays = [0.05, 0.10, 0.25, 0.80, 0.95, 1.0, 1.0]
    output = actions(SevenRayAutopilot(), rows(), rays)
    assert sum(action[0] == STEER_RIGHT for action in output) > 50
    assert any(action[2] == SHOULDER_R for action in output)
    assert all(action[1] == THROTTLE_ACCEL for action in output)


def test_right_obstacle_turns_and_leans_left():
    rays = [1.0, 1.0, 0.95, 0.80, 0.25, 0.10, 0.05]
    output = actions(SevenRayAutopilot(), rows(), rays)
    assert sum(action[0] == STEER_LEFT for action in output) > 50
    assert any(action[2] == SHOULDER_L for action in output)


def test_steering_bursts_are_proportional_to_ray_asymmetry():
    mild = actions(
        SevenRayAutopilot(), rows(), [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], 200
    )
    strong = actions(
        SevenRayAutopilot(), rows(), [0.2, 0.3, 0.4, 0.8, 0.9, 1.0, 1.0], 200
    )
    mild_turns = sum(action[0] == STEER_RIGHT for action in mild)
    strong_turns = sum(action[0] == STEER_RIGHT for action in strong)
    assert 0 < mild_turns < strong_turns


def test_closing_left_rays_trigger_anticipatory_right_turn():
    controller = SevenRayAutopilot()
    controller.update(rows(), [0.9] * 7, 1500)
    output = actions(controller, rows(), [0.3, 0.35, 0.5, 0.9, 0.9, 0.9, 0.9], 20)
    assert any(action[0] == STEER_RIGHT for action in output)


def test_offroad_recovery_commits_toward_road_and_keeps_accelerating():
    controller = SevenRayAutopilot()
    offroad = rows(left=0, right=0, on_road=False, nearest=220)
    output = actions(controller, offroad, [0.1] * 7, 20)
    assert all(action[0] == STEER_RIGHT for action in output)
    assert all(action[1] == THROTTLE_ACCEL for action in output)
    assert all(action[2] == SHOULDER_R for action in output)


def test_metrics_report_center_wall_response_and_seven_ray_clearance():
    controller = SevenRayAutopilot()
    metrics = AutopilotMetrics()
    readings = rows()
    rays = [0.3, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0]
    for _ in range(100):
        action, _ = controller.update(readings, rays, 1500)
        metrics.update(readings, rays, action[0], controller.last_ray_signal)
    summary = metrics.summary()
    assert summary["centering"] == 1.0
    assert summary["wall_contact"] == 0.0
    assert summary["ray_response"] > 0.5
    assert summary["forward_clearance"] > 0.65


def run():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    run()
