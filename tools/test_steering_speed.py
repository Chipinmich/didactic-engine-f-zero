"""Measure whether BizHawk host speed changes F-Zero steering authority.

Each speed mode gets a fresh, isolated emulator. From the same savestate the
script replays equal-length straight, left, and right inputs. The distance
between the left and right world-coordinate endpoints is the primary result:
it does not need a heading address and removes the track's natural curvature.
"""

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (  # noqa: E402
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    SHOULDER_NONE,
    BridgeClient,
    encode_action,
    instance_paths,
)


# Replays are expected to be bit-identical, so this is slack for the one
# stray repeat seen at 50% (9 of 10 identical, one 1.6% out), not a budget.
TOLERATED_SPREAD = 0.03

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIZHAWK_DIR = r"C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
EMUHAWK = os.path.join(BIZHAWK_DIR, "EmuHawk.exe")
ROM = os.path.join(BIZHAWK_DIR, "F-Zero (USA).sfc")
LUA_SCRIPT = os.path.join(BASE_DIR, "lua", "bridge.lua")


def distance(a, b):
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def wait_for_bridge(state_file, started, timeout=45.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if os.path.getmtime(state_file) >= started:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("diagnostic emulator did not start its bridge")


def endpoint(bridge, steer, frames, slot):
    bridge.reset(slot=slot)
    action = encode_action(steer, THROTTLE_ACCEL, SHOULDER_NONE)
    speed = 0
    for _ in range(frames):
        speed, _frame = bridge.advance(action)
    return tuple(bridge._position), speed


def measure_mode(mode, instance, frames, repeats, slot):
    action_file, state_file, _frame_file, _mmf_name = instance_paths(instance)
    for stale in (action_file, state_file):
        try:
            os.remove(stale)
        except OSError:
            pass

    env = dict(
        os.environ,
        FZERO_INSTANCE=str(instance),
        FZERO_START_SLOT=str(slot),
        FZERO_SPEEDMODE=str(mode),
        # This tool exists to measure the speed-mode difference, so it is the
        # one caller allowed to drive a bridge that is not at
        # REQUIRED_SPEED_MODE.
        FZERO_ALLOW_SPEEDMODE_MISMATCH="1",
    )
    started = time.time()
    process = subprocess.Popen(
        [EMUHAWK, ROM, f"--lua={LUA_SCRIPT}"],
        env=env,
        cwd=BIZHAWK_DIR,
    )
    os.environ["FZERO_ALLOW_SPEEDMODE_MISMATCH"] = "1"
    bridge = None
    try:
        wait_for_bridge(state_file, started)
        bridge = BridgeClient(instance=instance)
        rows = []
        for _ in range(repeats):
            straight, straight_speed = endpoint(
                bridge, STEER_STRAIGHT, frames, slot
            )
            left, left_speed = endpoint(bridge, STEER_LEFT, frames, slot)
            right, right_speed = endpoint(bridge, STEER_RIGHT, frames, slot)
            rows.append(
                {
                    "straight": straight,
                    "left": left,
                    "right": right,
                    "left_from_straight": distance(left, straight),
                    "right_from_straight": distance(right, straight),
                    "left_right_span": distance(left, right),
                    "mean_speed": float(np.mean((straight_speed, left_speed, right_speed))),
                }
            )
        return rows
    finally:
        if bridge is not None:
            bridge.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="25,50,100")
    parser.add_argument("--instance", type=int, default=14)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slot", type=int, default=2)
    parser.add_argument(
        "--require-deterministic", action="store_true",
        help="exit non-zero unless every mode replays identically; use with "
             "--modes set to the mode training actually runs at",
    )
    args = parser.parse_args()

    modes = [int(value) for value in args.modes.split(",")]
    results = {}
    print(
        f"Same slot, {args.frames} emulated input frames, "
        f"{args.repeats} repeats per speed mode\n"
    )
    for mode in modes:
        print(f"Testing BizHawk speed mode {mode}%...")
        rows = measure_mode(mode, args.instance, args.frames, args.repeats, args.slot)
        results[mode] = rows
        spans = np.asarray([row["left_right_span"] for row in rows])
        left = np.asarray([row["left_from_straight"] for row in rows])
        right = np.asarray([row["right_from_straight"] for row in rows])
        speeds = np.asarray([row["mean_speed"] for row in rows])
        print(
            f"  left/right span {spans.mean():.2f} +/- {spans.std():.2f}; "
            f"left {left.mean():.2f}; right {right.mean():.2f}; "
            f"final speed {speeds.mean():.1f}"
        )
        print("  repeat spans: " + ", ".join(f"{value:.2f}" for value in spans))

    baseline = float(np.mean([row["left_right_span"] for row in results[modes[0]]]))
    print("\nRelative steering authority (first mode = 100%):")
    for mode in modes:
        span = float(np.mean([row["left_right_span"] for row in results[mode]]))
        relative = 100.0 * span / baseline if baseline else float("nan")
        print(f"  {mode:>3}% speed mode: {relative:6.1f}%")

    if args.require_deterministic:
        # Determinism, not authority, is the property worth guarding. A
        # consistently weaker environment can be learned around; one that
        # answers differently every replay cannot, and the genetic search
        # ranks genomes on differences far smaller than the 100% spread.
        print("\nReplay determinism:")
        failed = []
        for mode in modes:
            spans = np.asarray([row["left_right_span"] for row in results[mode]])
            spread = float(spans.max() - spans.min())
            average = max(float(spans.mean()), 1e-9)
            ok = spread <= TOLERATED_SPREAD * average
            print(f"  {mode:>3}%: spread {spread:.2f} over {len(spans)} replays "
                  f"({spread / average:.1%}) -- {'ok' if ok else 'NOT REPRODUCIBLE'}")
            if not ok:
                failed.append(mode)
        if failed:
            print(
                "\nIdentical inputs from one savestate did not replay "
                f"identically at: {', '.join(str(m) + '%' for m in failed)}."
            )
            return 1
        print("\nEvery mode replayed identically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
