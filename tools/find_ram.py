"""Find the RAM addresses holding lap count, track progress and power.

Reads the capture produced by lua/ram_record.lua and looks for addresses
whose behaviour over the recording matches what each quantity must do:

  lap counter   a small integer that only ever steps up, by one, a handful
                of times across the whole run
  progress      rises steadily through a lap then snaps back to near zero,
                once per lap -- a sawtooth
  power/health  falls when you hit a wall and refills on the recharge strip,
                so it moves in both directions within a bounded range

Nothing here is guesswork about a specific ROM: candidates are ranked by how
well they fit those shapes, and you confirm the winners in BizHawk.

    python tools/find_ram.py
"""

import os
import sys

import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CAPTURE = os.path.join(BASE_DIR, "ram_capture.bin")
META = os.path.join(BASE_DIR, "ram_capture.csv")

WRAM_SIZE = 0x20000

MAX_LAPS_EXPECTED = 8
TOP_N = 12


def load():
    if not os.path.exists(CAPTURE):
        raise SystemExit(
            f"No capture at {CAPTURE}\n"
            "Load lua/ram_record.lua in BizHawk, drive 2-3 laps, stop it, "
            "then re-run this."
        )

    raw = np.fromfile(CAPTURE, dtype=np.uint8)
    samples = raw.size // WRAM_SIZE

    if samples < 20:
        raise SystemExit(
            f"Only {samples} samples captured -- not enough to analyse. "
            "Record a longer run (2-3 full laps)."
        )

    ram = raw[: samples * WRAM_SIZE].reshape(samples, WRAM_SIZE)
    return ram


def as_words(ram):
    """Little-endian 16-bit view, since counters often span two bytes."""
    low = ram[:, 0::2].astype(np.uint16)
    high = ram[:, 1::2].astype(np.uint16)
    return low | (high << 8)


def find_lap_counters(ram):
    """Small integers that only step upward, a few times, by one."""
    results = []

    diffs = np.diff(ram.astype(np.int16), axis=0)
    never_decreases = (diffs >= 0).all(axis=0)
    steps = (diffs > 0).sum(axis=0)
    span = ram.max(axis=0).astype(int) - ram.min(axis=0).astype(int)

    candidates = np.flatnonzero(
        never_decreases
        & (steps >= 1)
        & (steps <= MAX_LAPS_EXPECTED)
        & (span >= 1)
        & (span <= MAX_LAPS_EXPECTED)
        & (ram.max(axis=0) <= 16)
    )

    for addr in candidates:
        column = ram[:, addr]
        increments = diffs[:, addr]
        # A lap counter goes up by exactly one at a time.
        if increments[increments > 0].max() != 1:
            continue
        results.append((int(steps[addr]), int(addr), column.min(), column.max()))

    results.sort(key=lambda r: -r[0])
    return results


def find_progress(words, laps_seen):
    """Sawtooth: climbs through a lap, resets sharply, repeats."""
    results = []

    values = words.astype(np.float64)
    diffs = np.diff(values, axis=0)

    rises = (diffs > 0).sum(axis=0)
    falls = diffs < 0
    fall_count = falls.sum(axis=0)

    spans = values.max(axis=0) - values.min(axis=0)

    # Mostly rising, with only a few sharp drops -- one per lap.
    plausible = np.flatnonzero(
        (spans > 200)
        & (rises > values.shape[0] * 0.5)
        & (fall_count >= 1)
        & (fall_count <= MAX_LAPS_EXPECTED * 2)
    )

    for idx in plausible:
        column = values[:, idx]
        drops = diffs[:, idx][diffs[:, idx] < 0]
        if drops.size == 0:
            continue

        # The reset should dwarf the ordinary per-sample increase.
        typical_rise = np.median(diffs[:, idx][diffs[:, idx] > 0])
        biggest_drop = -drops.min()
        if typical_rise <= 0 or biggest_drop < typical_rise * 20:
            continue

        score = biggest_drop / typical_rise
        results.append((score, idx * 2, int(column.min()), int(column.max()), int(fall_count[idx])))

    results.sort(key=lambda r: -r[0])
    return results


# A health value moves gradually and sits still between events, whereas an
# arbitrary byte churns. Ranking by range alone drowns the real thing in
# noise -- span divided by mean step separates them by an order of
# magnitude, so rank by that instead.
MIN_SMOOTHNESS = 5.0


def find_progress_by_laps(values, lap_values, label, min_span=40, min_monotonic=0.9):
    """Progress search that uses known lap boundaries.

    Far more reliable than looking for a sawtooth shape blind: once the lap
    counter says where each lap starts, progress is simply whatever climbs
    steadily through a lap and is back near the bottom at the start of the
    next one.
    """
    boundaries = np.flatnonzero(np.diff(lap_values.astype(np.int32)) > 0) + 1
    if boundaries.size == 0:
        return []

    segments = [s for s in np.split(np.arange(values.shape[0]), boundaries) if s.size >= 5]
    if len(segments) < 2:
        return []

    n = values.shape[1]
    monotonic = np.zeros(n)
    spans = np.zeros(n)
    restarts = np.zeros(n)

    for segment in segments:
        block = values[segment].astype(np.float64)
        diffs = np.diff(block, axis=0)
        monotonic += (diffs >= 0).mean(axis=0)
        spans += block.max(axis=0) - block.min(axis=0)

    monotonic /= len(segments)
    spans /= len(segments)

    # Each lap should start low again rather than carrying on upward.
    for previous, current in zip(segments, segments[1:]):
        restarts += (values[current[0]] < values[previous[-1]] * 0.5).astype(float)
    restarts /= max(len(segments) - 1, 1)

    candidates = np.flatnonzero(
        (monotonic >= min_monotonic) & (spans >= min_span) & (restarts >= 0.5)
    )

    results = []
    stride = 2 if label == "16-bit" else 1
    for idx in candidates:
        results.append(
            (
                float(monotonic[idx] * spans[idx]),
                int(idx * stride),
                float(monotonic[idx]),
                int(spans[idx]),
            )
        )

    results.sort(key=lambda r: -r[0])
    return results


def find_power(ram, speeds):
    """Bounded value that falls and recovers gradually, and isn't the speed."""
    results = []

    values = ram.astype(np.float64)
    diffs = np.diff(values, axis=0)

    ups = (diffs > 0).sum(axis=0)
    downs = (diffs < 0).sum(axis=0)
    spans = values.max(axis=0) - values.min(axis=0)
    steadiness = (diffs == 0).mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_step = np.abs(diffs).mean(axis=0)
        smoothness = np.where(mean_step > 0, spans / mean_step, 0.0)

    plausible = np.flatnonzero(
        (ups >= 2)
        & (downs >= 2)
        & (spans >= 20)
        & (values.max(axis=0) <= 255)
        & (smoothness >= MIN_SMOOTHNESS)
        & (steadiness >= 0.2)
    )

    for addr in plausible:
        column = values[:, addr]
        # Power should not simply track speed.
        if speeds is not None and speeds.size == column.size:
            if np.std(column) > 0 and np.std(speeds) > 0:
                correlation = abs(np.corrcoef(column, speeds)[0, 1])
                if correlation > 0.8:
                    continue
        results.append(
            (float(smoothness[addr]), int(addr), int(column.min()), int(column.max()))
        )

    results.sort(key=lambda r: -r[0])
    return results


def load_speeds(samples):
    if not os.path.exists(META):
        return None
    try:
        data = np.genfromtxt(META, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        speeds = data[:, 2]
        return speeds[:samples] if speeds.size >= samples else None
    except (ValueError, IndexError):
        return None


def main():
    ram = load()
    samples = ram.shape[0]
    speeds = load_speeds(samples)

    print(f"{samples} samples of {WRAM_SIZE} bytes\n")

    print("=== lap counter candidates ===")
    laps = find_lap_counters(ram)
    if not laps:
        print("  none found -- did the recording cover a lap boundary?")
    for steps, addr, lo, hi in laps[:TOP_N]:
        print(f"  0x{addr:05X}  stepped up {steps}x, range {lo}..{hi}")

    print()
    print("=== progress / checkpoint candidates ===")

    if laps:
        lap_addr = laps[0][1]
        lap_values = ram[:, lap_addr]
        print(f"  (segmented by lap counter at 0x{lap_addr:05X})")

        for label, view in (("8-bit", ram), ("16-bit", as_words(ram))):
            found = find_progress_by_laps(view, lap_values, label)
            if not found:
                print(f"  {label}: none")
                continue
            for score, addr, mono, span in found[:TOP_N // 2]:
                print(
                    f"  {label} 0x{addr:05X}  rises {mono*100:5.1f}% of the "
                    f"time, ~{span} per lap"
                )
    else:
        progress = find_progress(as_words(ram), 0)
        if not progress:
            print("  none found -- needs a lap counter or a full lap")
        for score, addr, lo, hi, resets in progress[:TOP_N]:
            print(
                f"  0x{addr:05X}  sawtooth score {score:8.0f}, "
                f"range {lo}..{hi}, {resets} reset(s)"
            )

    print()
    print("=== power / health candidates ===")
    power = find_power(ram, speeds)
    if not power:
        print("  none found -- try a run where you scrape a wall and then")
        print("  cross the recharge strip")
    for smooth, addr, lo, hi in power[:TOP_N]:
        print(f"  0x{addr:05X}  range {lo}..{hi}, smoothness {smooth:6.1f}")

    print()
    print("Confirm the top candidates in BizHawk (Tools > RAM Watch) before")
    print("wiring them in: drive a lap and check each one behaves as labelled.")


if __name__ == "__main__":
    main()
