"""Build a track centreline from a recorded human lap, for measuring progress.

WHY THIS EXISTS. The bridge used to report "progress" from RAM 0x000C1, and
roughly 40,000 training episodes were scored on it. 0x000C1 is the LAP TIMER.
On a parked car mid-race it counted 18 -> 26 over 124 frames while the speed
reading sat at exactly 0 and the integral of speed never moved; it ticks on a
fixed ~15-frame cadence whatever the car is doing.

It survived the original search because a lap timer and a lap position have the
SAME sawtooth shape -- rises through the lap, resets at the line, never
decreases -- so ranking addresses by shape cannot tell them apart. The test that
can is holding the car still in mid-race: position freezes, a clock does not.

F-Zero stores no lap-position counter at all. A sweep of all 131072 work-RAM
addresses for a per-lap sawtooth that ALSO freezes with the car returned only
the timer. What it does store is the machine's world position:

    x = 0x00B70 (u16 LE)      y = 0x00B90 (u16 LE)

verified on a human recording of four laps. Picking the pair matters: several
nearby addresses also track position but are redundant with each other
(|corr| = 0.999), and such a pair traces a line out and back rather than a loop,
folding the track onto itself. The test that separates them is physical -- for
true coordinates the per-frame displacement magnitude is proportional to speed.
This pair scores corr(|dp|, speed) = +0.9922 with corr(x, y) = +0.385; every
other candidate pair scored 0.28.

So progress is derived rather than read: this script resamples one recorded lap
into equally spaced checkpoints, and track_map.py maps a live (x, y) onto the
nearest one.

    python tools/build_track_map.py
"""

import csv
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

CAPTURE = os.path.join(BASE_DIR, "ram_capture.bin")
META = os.path.join(BASE_DIR, "ram_capture.csv")
OUT = os.path.join(BASE_DIR, "models", "track_mute_city.npz")

WRAM_SIZE = 0x20000
X_ADDRESS = 0x00B70
Y_ADDRESS = 0x00B90
# The lap counter is 0x00F53 (0x01161 mirrors it), NOT 0x000C0.
#
# 0x000C0/0x000C1 are a two-field packed-BCD CLOCK: 0x000C1 runs 0..89 and
# carries into 0x000C0. On the recording that carry happened at frame 5135,
# twenty frames from a genuine lap boundary at 5155 -- close enough to look
# like a lap counter and build a map from two laps instead of one, which
# aliased the centreline onto itself 2:1 (checkpoints 4 and 49 landed at the
# same coordinates).
#
# 0x00F53 ticks at frames 3255, 5155, 6800 and 8515, and the machine returns
# to its lap-start position at 3235, 5140, 6785 and 8500 -- agreeing within
# ~20 frames on all four.
LAP_ADDRESS = 0x00F53

# Checkpoints per lap. 90 keeps the units the rest of the project already
# reasons in ("reaches checkpoint 83 of 90") and is fine enough that a
# checkpoint is a couple of machine lengths.
CHECKPOINTS = 90


def read_u16(ram, address):
    return ram[:, address].astype(np.int32) + 256 * ram[:, address + 1].astype(np.int32)


def load():
    if not os.path.exists(CAPTURE):
        raise SystemExit(
            f"No capture at {CAPTURE}\n"
            "Load lua/ram_record.lua in BizHawk, drive at least one complete "
            "lap, stop the script, then re-run this."
        )
    raw = np.fromfile(CAPTURE, dtype=np.uint8)
    ram = raw[: (raw.size // WRAM_SIZE) * WRAM_SIZE].reshape(-1, WRAM_SIZE)
    rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(open(META))]
    count = min(len(ram), len(rows))
    return ram[:count], rows[:count]


def longest_complete_lap(lap_counter, speed):
    """One full lap: between two lap ticks, or from the grid to the first tick.

    A capture starts on the title/attract screen, where the demo drives at
    racing speed -- so "first frame with speed > 0" lands in the demo, not on
    the player's lap, and building from there gave a 19.9% start/finish gap.
    The player's lap begins at the END of the standing start instead: the last
    stationary sample before the first lap tick.
    """
    changes = [i for i in range(1, len(lap_counter))
               if lap_counter[i] != lap_counter[i - 1]]
    if len(changes) >= 2:
        spans = list(zip(changes, changes[1:]))
        return max(spans, key=lambda s: s[1] - s[0])
    if not changes:
        raise SystemExit(
            "no lap tick in the capture -- 0x00F53 never changed, so no "
            "complete lap was driven"
        )
    end = changes[0]
    stationary = np.where(speed[:end] == 0)[0]
    if not len(stationary):
        raise SystemExit("no standing start found before the first lap tick")
    start = int(stationary[-1])
    if end - start < 50:
        raise SystemExit("no complete lap in the capture")
    return start, end


def resample(points, count):
    """Equally spaced by arc length, so checkpoints are equally spaced on track."""
    step = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0:
        raise SystemExit("the recorded lap has zero length")
    wanted = np.linspace(0.0, arc[-1], count, endpoint=False)
    return np.stack([np.interp(wanted, arc, points[:, axis]) for axis in (0, 1)], 1)


def main():
    ram, rows = load()
    speed = np.array([r["speed"] for r in rows])
    x, y = read_u16(ram, X_ADDRESS), read_u16(ram, Y_ADDRESS)
    laps = ram[:, LAP_ADDRESS].astype(np.int32)

    start, end = longest_complete_lap(laps, speed)
    print(f"using samples {start}..{end} "
          f"(frames {rows[start]['frame']}..{rows[end]['frame']}, lap {laps[start]})")

    path = np.stack([x[start:end], y[start:end]], 1).astype(np.float64)
    centreline = resample(path, CHECKPOINTS)

    span = np.hypot(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
    closure = float(np.hypot(*(path[-1] - path[0])))
    print(f"track scale {span:.0f}, start/finish closure {closure:.0f} "
          f"({closure / span:.2%})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        centreline=centreline.astype(np.float32),
        checkpoints=CHECKPOINTS,
        x_address=X_ADDRESS,
        y_address=Y_ADDRESS,
    )
    print(f"wrote {OUT}: {CHECKPOINTS} checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
