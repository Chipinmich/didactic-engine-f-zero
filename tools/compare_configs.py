"""Race several controller configurations against each other, repeatedly.

Tuning against single runs does not work here: identical configurations
have scored 81 then 66, and 96 then 54. Differences that size are
run-to-run variance, and reading them as improvements is how a completed
lap once got claimed that never happened.

So every configuration is run several times and reported as a mean with a
spread. If the spreads overlap, the difference is not real.

    python tools/launch_instances.py 4
    python tools/compare_configs.py --repeats 3 --seconds 180

Configurations rotate across emulators between rounds, so one slow or
unlucky instance cannot consistently flatter the same configuration.
"""

import argparse
import os
import statistics
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import ACTION_NONE, BridgeClient, BridgeTimeout, encode_action
from control import SCAN_FRACTIONS, Controller
from vision.hud import PowerTracker
from vision.road_edges import measure_rows

CRASH_FRAMES = 30
MIN_ROAD_PIXELS = 120

# Current defaults, tuned earlier on distance and wall contact.
BEST = {
    "deadzone": 0.04,
    "full_scale": 0.35,
    "curve_slowdown": 1.9,
    "min_corner_speed": 800,
}

CONFIGS = [
    ("current", dict(BEST)),
    ("early-lookahead", dict(BEST, row_weights=(0.8, 0.9, 0.7))),
    ("gentle-corners", dict(BEST, curve_slowdown=2.6, min_corner_speed=700)),
    ("wide-margin", dict(BEST, target_clearance=85)),
]


class Trial:
    """One run of one configuration."""

    def __init__(self):
        self.furthest = 0
        self.covered = 0
        self.crashes = 0
        self.laps = 0
        self.frames = 0
        self.speed_total = 0.0
        self.wall_frames = 0
        self.power_used = 0.0
        self.error = None

    @property
    def mean_speed(self):
        return self.speed_total / max(self.frames, 1)

    @property
    def wall_contact(self):
        return self.wall_frames / max(self.frames, 1)

    @property
    def power_per_cp(self):
        """Power burned per checkpoint -- the real cost of a driving style.

        Power is what ends a run, and dividing by distance makes a config
        that dies early comparable with one that gets further.
        """
        return self.power_used / max(self.covered, 1)


def drive(instance, settings, seconds, trial):
    bridge = BridgeClient(instance=instance)
    controller = Controller(**settings)

    # Created before the try, because the finally block reports from it --
    # a failure in reset() would otherwise raise NameError and hide the
    # actual error.
    power = PowerTracker()

    try:
        bridge.reset()
        start_total = bridge.total_progress
        start_lap = bridge.lap
        covered_this_life = 0
        blind = 0
        action = ACTION_NONE
        deadline = time.time() + seconds

        while time.time() < deadline:
            speed, frame = bridge.advance(action)
            if frame is None:
                continue

            trial.frames += 1
            trial.speed_total += speed

            h, w = frame.shape[:2]
            readings = measure_rows(frame, [int(h * f) for f in SCAN_FRACTIONS])

            if sum(r["road_pixels"] for r in readings) >= MIN_ROAD_PIXELS:
                blind = 0
            else:
                blind += 1
                if blind >= CRASH_FRAMES:
                    trial.crashes += 1
                    bridge.reset()
                    controller = Controller(**settings)
                    start_total = bridge.total_progress
                    start_lap = bridge.lap
                    covered_this_life = 0
                    power.reset()
                    blind = 0
                    action = ACTION_NONE
                    continue

            power.update(frame)

            gained = bridge.total_progress - start_total
            if gained > covered_this_life:
                # Counted for every life: ground re-driven after a crash
                # still costs power, so leaving it out inflates the drain
                # figure several times over.
                trial.covered += gained - covered_this_life
                covered_this_life = gained

            trial.furthest = max(trial.furthest, gained)

            # Laps SINCE THIS LIFE STARTED. The game's absolute lap counter
            # reads as a completed lap whenever it happens to be non-zero.
            trial.laps = max(trial.laps, bridge.lap - start_lap)

            near = readings[0]
            if near["left_px"] == 0 or near["right_px"] == 0:
                trial.wall_frames += 1

            dims, _ = controller.update(readings, w / 2.0, speed)
            action = encode_action(*dims)

    except BridgeTimeout as exc:
        trial.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - report, don't kill the comparison
        trial.error = f"{type(exc).__name__}: {exc}"
    finally:
        trial.power_used = power.drained
        bridge.close()


def spread(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--instances", type=int, default=len(CONFIGS))
    args = parser.parse_args()

    trials = {name: [] for name, _ in CONFIGS}

    print(f"{len(CONFIGS)} configs x {args.repeats} runs of {args.seconds:.0f}s")
    print(f"about {args.repeats * args.seconds / 60:.0f} minutes\n")

    for round_index in range(args.repeats):
        print(f"round {round_index + 1}/{args.repeats}...")
        threads = []
        round_trials = []

        for i, (name, settings) in enumerate(CONFIGS):
            instance = (i + round_index) % args.instances
            trial = Trial()
            trials[name].append(trial)
            round_trials.append((name, trial))

            thread = threading.Thread(
                target=drive,
                args=(instance, settings, args.seconds, trial),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join(timeout=args.seconds + 90)

        for name, trial in round_trials:
            note = f"  ERROR {trial.error}" if trial.error else ""
            print(
                f"    {name:16s} furthest {trial.furthest:3d}  "
                f"covered {trial.covered:3d}  crashes {trial.crashes}{note}"
            )

    print()
    print(
        f"{'config':16s} {'furthest':>13} {'covered':>13} {'crashes':>11} "
        f"{'pwr/cp':>13} {'wall%':>7} {'laps':>5}"
    )

    rows = []
    for name, runs in trials.items():
        rows.append(
            (
                statistics.mean([t.covered for t in runs]),
                name,
                [t.furthest for t in runs],
                [t.covered for t in runs],
                [t.crashes for t in runs],
                [t.power_per_cp * 100 for t in runs],
                [t.wall_contact * 100 for t in runs],
                max(t.laps for t in runs),
            )
        )

    for mean_covered, name, furthest, covered, crashes, power, wall, laps in sorted(
        rows, reverse=True
    ):
        print(
            f"{name:16s} "
            f"{statistics.mean(furthest):5.0f} +-{spread(furthest):4.0f}  "
            f"{mean_covered:5.0f} +-{spread(covered):4.0f}  "
            f"{statistics.mean(crashes):4.1f} +-{spread(crashes):3.1f}  "
            f"{statistics.mean(power):6.1f} +-{spread(power):4.1f}  "
            f"{statistics.mean(wall):5.1f}%  {laps:4d}"
        )

    print()
    print("Differences smaller than the +- spread are not real.")


if __name__ == "__main__":
    main()
