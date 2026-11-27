"""Offline tests for the seven forward road rays."""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vision.road_edges import (
    NUM_FEATURES,
    NUM_RAY_FEATURES,
    RAY_START_INDEX,
    SPEED_INDEX,
    draw_debug_overlay,
    features_from_readings,
    measure_rays,
    ray_distances,
)


def straight_road():
    frame = np.zeros((224, 256, 3), dtype=np.uint8)
    # Low-saturation mid-value gray is inside the calibrated road range.
    polygon = np.array([[25, 151], [96, 0], [160, 0], [231, 151]], np.int32)
    cv2.fillConvexPoly(frame, polygon, (150, 150, 150))
    return frame


def test_seven_rays_fan_forward_from_the_car():
    rays = measure_rays(straight_road())
    assert len(rays) == NUM_RAY_FEATURES == 7
    assert [ray["angle"] for ray in rays] == [-60, -40, -20, 0, 20, 40, 60]
    assert all(ray["origin"] == (128, 150) for ray in rays)
    assert all(ray["end"][1] <= ray["origin"][1] for ray in rays)
    assert rays[3]["distance"] > rays[0]["distance"]
    assert rays[3]["distance"] > rays[-1]["distance"]


def test_ray_features_are_normalised_and_mirror_symmetrically():
    frame = straight_road()
    original = ray_distances(frame)
    mirrored = ray_distances(cv2.flip(frame, 1))
    assert original.shape == (7,)
    assert np.all((0.0 <= original) & (original <= 1.0))
    assert np.allclose(original, mirrored[::-1], atol=0.04)


def test_non_road_frame_reports_seven_immediate_obstacles():
    frame = np.zeros((224, 256, 3), dtype=np.uint8)
    assert np.allclose(ray_distances(frame), 0.0)


def test_observation_keeps_speed_and_appends_all_rays():
    values = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    observation = features_from_readings([], 0.42, values)
    assert observation.shape == (NUM_FEATURES,)
    assert observation[SPEED_INDEX] == np.float32(0.42)
    assert np.allclose(observation[RAY_START_INDEX:], values)


def test_debug_overlay_draws_the_ray_fan_without_changing_shape():
    frame = straight_road()
    overlay = draw_debug_overlay(frame, [], measure_rays(frame))
    assert overlay.shape == frame.shape
    assert np.any(overlay != frame)


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
