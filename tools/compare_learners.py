"""Compare the genetic search and the DQN on the same task, honestly.

The two learn very differently -- a population search evaluates whole genomes
over fixed episodes, a gradient method updates continuously -- so per-episode
or per-update counts are not comparable. What IS comparable is what they
achieve for the same wall-clock time on the same number of emulators, which is
why tools/overnight_training.py now splits the instances evenly.

Progress is reported in checkpoints gained on the derived track map, the same
measure for both. Episodes are bucketed by elapsed minutes since each run
began rather than by index, so a learner producing shorter episodes is not
credited for producing more of them.

    python tools/compare_learners.py
    python tools/compare_learners.py --ga neuro_20260821_202154 --dqn dqn_v11_compare
"""

import argparse
import glob
import json
import os
import statistics as st
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

STORE = os.path.join(BASE_DIR, "runs", "store")


def load(run_id):
    rows = []
    for path in glob.glob(os.path.join(STORE, run_id, "episodes.*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def newest(prefix):
    """Most recently written run, by mtime -- NOT by name.

    Sorting names alphabetically picked up "neuro_smoke", an old run with a
    different field set, in preference to the live timestamped one.
    """
    candidates = [
        d for d in os.listdir(STORE)
        if d.startswith(prefix) and os.path.isdir(os.path.join(STORE, d))
    ]
    if not candidates:
        return None

    def written(name):
        """Newest episode file inside, NOT the directory's own mtime.

        A directory's mtime only changes when an entry is added or removed, so
        a run that has been appending to the same episodes.*.jsonl for hours
        still looks older than one created a minute ago. That silently picked
        a stale run for comparison.
        """
        files = glob.glob(os.path.join(STORE, name, "episodes.*.jsonl"))
        if not files:
            return os.path.getmtime(os.path.join(STORE, name))
        return max(os.path.getmtime(f) for f in files)

    return max(candidates, key=written)


def elapsed_minutes(rows):
    """Approximate minutes into the run, per episode.

    Nothing records a timestamp, but every episode records its own duration in
    `seconds`, and each emulator instance runs its episodes back to back. So
    the running total of `seconds` within an instance tracks elapsed time on
    that emulator. Both learners were given the same number of emulators for
    the same wall-clock window, which is what makes this comparable.
    """
    running = {}
    minutes = []
    usable = False
    for row in rows:
        instance = row.get("instance", 0)
        seconds = row.get("seconds")
        if not isinstance(seconds, (int, float)):
            minutes.append(0.0)
            continue
        usable = True
        running[instance] = running.get(instance, 0.0) + float(seconds)
        minutes.append(running[instance] / 60.0)
    return minutes, usable


def summarise(label, rows, buckets, timed):
    if not rows:
        print(f"{label}: no episodes recorded")
        return
    rows = [r for r in rows if "progress_gain" in r]
    if not rows:
        print(f"{label}: episodes recorded without progress_gain (old format)")
        return
    gains = [r["progress_gain"] for r in rows]
    best = max(gains)
    reach = max(r.get("start_progress", 0) + r["progress_gain"] for r in rows)
    print(f"{label}")
    print(f"  episodes {len(rows)}   mean gain {st.mean(gains):.2f}   "
          f"median {st.median(gains):.0f}   best {best}")
    print(f"  furthest checkpoint reached {reach:.0f} of 90")
    print(f"  mean speed {st.mean(r['mean_speed'] for r in rows):.0f}   "
          f"wall {st.mean(r['wall_rate'] for r in rows):.1%}   "
          f"steps/ep {st.mean(r['steps'] for r in rows):.0f}")
    if buckets and timed:
        print("  by elapsed minutes:")
        for lo, hi, values in buckets:
            if values:
                print(f"    {lo:>3.0f}-{hi:<3.0f} min  n={len(values):5d}  "
                      f"mean {st.mean(values):5.2f}  best {max(values):3.0f}")


def bucket(rows, minutes, width):
    out = []
    if not minutes:
        return out
    top = max(minutes)
    lo = 0.0
    while lo <= top:
        values = [
            r["progress_gain"] for r, m in zip(rows, minutes)
            if lo <= m < lo + width and "progress_gain" in r
        ]
        out.append((lo, lo + width, values))
        lo += width
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga", default=None, help="genetic run id (default: newest)")
    parser.add_argument("--dqn", default=None, help="DQN run id (default: newest)")
    parser.add_argument("--ppo", default=None, help="PPO run id (default: newest)")
    parser.add_argument("--bucket-minutes", type=float, default=15.0)
    args = parser.parse_args()

    wanted = [
        ("GENETIC", args.ga or newest("neuro_")),
        ("DQN", args.dqn or newest("dqn_overnight")),
        ("PPO", args.ppo or newest("ppo_overnight")),
    ]
    found = [(label, run_id) for label, run_id in wanted if run_id]
    if not found:
        raise SystemExit("no runs found; pass --ga / --dqn / --ppo")
    for label, run_id in wanted:
        print(f"{label:8s} {run_id or '(none found)'}")
    print("Same track map, same checkpoints, same emulator count.\n")

    results = {}
    for label, run_id in found:
        rows = load(run_id)
        minutes, timed = elapsed_minutes(rows)
        summarise(label, rows, bucket(rows, minutes, args.bucket_minutes), timed)
        results[label] = rows
        print()
        if not timed:
            print("  (no timestamps recorded, so time buckets are unavailable)\n")

    live = {k: v for k, v in results.items() if v}
    if len(live) > 1:
        print("HEAD TO HEAD")
        header = "".join(f"{k:>12s}" for k in live)
        print(f"  {'':14s}{header}{'':>4s}leader")
        for name, f, higher_is_better in (
            ("mean gain", lambda r: st.mean(x["progress_gain"] for x in r), True),
            ("median gain", lambda r: float(st.median(x["progress_gain"] for x in r)), True),
            ("best gain", lambda r: float(max(x["progress_gain"] for x in r)), True),
            ("mean speed", lambda r: st.mean(x["mean_speed"] for x in r), True),
            ("wall rate", lambda r: st.mean(x["wall_rate"] for x in r), False),
        ):
            values = {k: f(v) for k, v in live.items()}
            pick = max if higher_is_better else min
            leader = pick(values, key=values.get)
            row = "".join(f"{values[k]:>12.2f}" for k in live)
            print(f"  {name:14s}{row}    {leader}")
        print()
        print("Read this as a snapshot, not a verdict: this project has measured")
        print("identical configurations at 81 and 66, so a single comparison at")
        print("one point in training is weak evidence. Re-run it as both mature.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
