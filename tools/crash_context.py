"""Show what the controller was doing in the run-up to each crash.

A crash shows up in the telemetry as the road vanishing (both near edges
zero and staying there) or progress resetting. This prints the frames
leading in, so the cause is visible rather than guessed at.

    python tools/crash_context.py runs/run5.csv
"""

import argparse

import numpy as np

LEAD_IN_FRAMES = 90
SAMPLE_EVERY = 10


def find_crashes(data):
    """Frame indices where progress resets backwards -- a restart."""
    progress = data["progress"]
    lap = data["lap"]
    total = lap * 90 + progress

    drops = np.flatnonzero(np.diff(total) < -20) + 1
    return drops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()

    data = np.genfromtxt(args.path, delimiter=",", names=True)
    crashes = find_crashes(data)

    if not len(crashes):
        print("no crashes found in this telemetry")
        return

    print(f"{len(crashes)} crash(es)\n")

    for n, at in enumerate(crashes, 1):
        start = max(at - LEAD_IN_FRAMES, 0)
        print(f"=== crash {n}: reset at frame {at}, cp {int(data['progress'][at-1])} ===")
        print("  frame   cp  speed target  sever  signal  steer thr shoul  nearL nearR  farL farR")

        for i in range(start, at, SAMPLE_EVERY):
            row = data[i]

            def fmt(value):
                return "  --" if np.isnan(value) else f"{int(value):4d}"

            print(
                f"  {int(row['frame']):5d} {int(row['progress']):4d} "
                f"{int(row['speed']):5d} {int(row['target']):6d} "
                f"{row['severity']:6.2f} {row['signal']:7.2f} "
                f"{int(row['steer']):5d} {int(row['throttle']):3d} {int(row['shoulder']):5d} "
                f"{fmt(row['near_left'])} {fmt(row['near_right'])} "
                f"{fmt(row['far_left'])} {fmt(row['far_right'])}"
            )

        window = data[start:at]
        print(
            f"  -> mean speed {window['speed'].mean():.0f}, "
            f"mean target {window['target'].mean():.0f}, "
            f"max severity {window['severity'].max():.2f}, "
            f"off-throttle {(window['throttle'] == 0).mean() * 100:.0f}%"
        )
        print()


if __name__ == "__main__":
    main()
