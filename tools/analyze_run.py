"""Find where a run loses time, from the telemetry autopilot.py --log writes.

    python tools/analyze_run.py runs/run1.csv

Reports the slowest stretches of track, how long each checkpoint took, and
what the controller was doing there -- speed against its own target, how
much it coasted, how hard it was steering, and whether it was stuck.
"""

import argparse
import os
import sys

import numpy as np

STUCK_SPEED = 700


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no telemetry at {path}")

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise SystemExit("telemetry file is empty")
    return np.atleast_1d(data)


def summarise(data):
    speed = data["speed"]
    target = data["target"]
    throttle = data["throttle"]
    steer = data["steer"]
    severity = data["severity"]

    frames = len(speed)
    coasting = (throttle == 0).mean()
    steering = (steer != 0).mean()

    print(f"frames            : {frames}")
    print(f"speed  mean/max   : {speed.mean():.0f} / {speed.max():.0f}")
    print(f"target mean       : {target.mean():.0f}")
    print(f"off throttle      : {coasting * 100:.1f}% of frames")
    print(f"steering          : {steering * 100:.1f}% of frames")
    print(f"severity mean/max : {severity.mean():.2f} / {severity.max():.2f}")
    print(f"below {STUCK_SPEED}        : {(speed < STUCK_SPEED).mean() * 100:.1f}% of frames")

    # Steering reversals per second: the signature of oscillation.
    turning = steer[steer != 0]
    if turning.size > 1:
        reversals = (np.diff(turning) != 0).sum()
        seconds = data["elapsed"][-1] - data["elapsed"][0]
        if seconds > 0:
            print(f"steer reversals   : {reversals / seconds:.1f}/s")


def per_checkpoint(data):
    """Frames spent on each checkpoint, worst first."""
    progress = data["progress"].astype(int)
    lap = data["lap"].astype(int)
    speed = data["speed"]
    throttle = data["throttle"]
    severity = data["severity"]

    key = lap * 1000 + progress
    rows = []

    start = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            block = slice(start, i)
            rows.append(
                (
                    i - start,
                    int(lap[start]),
                    int(progress[start]),
                    speed[block].mean(),
                    (throttle[block] == 0).mean(),
                    severity[block].mean(),
                )
            )
            start = i

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--worst", type=int, default=12)
    args = parser.parse_args()

    data = load(args.path)

    print("=== overall ===")
    summarise(data)

    rows = per_checkpoint(data)
    if not rows:
        return

    lengths = np.array([r[0] for r in rows])
    print()
    print("=== checkpoints ===")
    print(f"reached           : {len(rows)}")
    print(f"frames per cp     : median {np.median(lengths):.0f}, mean {lengths.mean():.0f}")

    print()
    print(f"=== slowest {args.worst} checkpoints ===")
    print("frames  lap  cp   mean speed   coast%   severity")
    for frames, lap, cp, spd, coast, sev in sorted(rows, reverse=True)[: args.worst]:
        print(f"{frames:6d}  {lap:3d}  {cp:3d}   {spd:9.0f}   {coast*100:5.1f}%   {sev:6.2f}")

    print()
    print("=== fastest 5 (for comparison) ===")
    print("frames  lap  cp   mean speed   coast%   severity")
    for frames, lap, cp, spd, coast, sev in sorted(rows)[:5]:
        print(f"{frames:6d}  {lap:3d}  {cp:3d}   {spd:9.0f}   {coast*100:5.1f}%   {sev:6.2f}")


if __name__ == "__main__":
    main()
