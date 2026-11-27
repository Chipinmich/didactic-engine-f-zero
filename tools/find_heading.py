"""Find the RAM address holding the machine's heading.

Lap and progress were found by recording RAM and looking for a value with
the right SHAPE -- one that only counts up, one that sawtooths. Heading has
no such shape: it wraps, and it moves both ways.

So this drives a scripted steering pattern instead. Because the bridge is
lockstep, the exact input on every frame is known, and heading is simply
whatever moves one way while turning left and the other way while turning
right. Nothing else in memory should do that.

    python tools/find_heading.py

Run it with one emulator free (it uses instance 9 by default).
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (
    BASE_DIR,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    SHOULDER_NONE,
    BridgeClient,
    encode_action,
)

BIZHAWK_DIR = r"C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
EMUHAWK = os.path.join(BIZHAWK_DIR, "EmuHawk.exe")
ROM = os.path.join(BIZHAWK_DIR, "F-Zero (USA).sfc")
LUA_SCRIPT = os.path.join(BASE_DIR, "lua", "bridge.lua")

WRAM_SIZE = 0x20000
DUMP_EVERY = 2

# Short alternating segments, NOT long ones. Holding a direction for 90
# frames simply drives into a wall, and a wrecked machine stops responding
# to the wheel -- which produced 390 "turning" samples of a car that was
# mostly dead, and agreement no better than a coin flip. Flicking left and
# right keeps it near the middle of the road and alive.
SEGMENT_FRAMES = 16
CYCLES = 24
SPOOL_FRAMES = 90


def launch(instance, dump_path):
    env = dict(
        os.environ,
        FZERO_INSTANCE=str(instance),
        FZERO_RAM_DUMP=dump_path,
        FZERO_RAM_EVERY=str(DUMP_EVERY),
    )
    process = subprocess.Popen(
        [EMUHAWK, ROM, f"--lua={LUA_SCRIPT}"], env=env, cwd=BIZHAWK_DIR
    )
    return process


def record(bridge):
    """Flick left and right, returning the steering sign and speed per sample.

    Speed comes back too so samples where the machine is wrecked -- and so
    ignoring the wheel entirely -- can be thrown away rather than diluting
    the signal.
    """
    signs = []
    speeds = []

    def hold(steer, frames, sign):
        action = encode_action(steer, THROTTLE_ACCEL, SHOULDER_NONE)
        for _ in range(frames):
            speed, _ = bridge.advance(action)
            if bridge._frame_count % DUMP_EVERY == 0:
                signs.append(sign)
                speeds.append(speed)

    bridge.reset()
    hold(STEER_STRAIGHT, SPOOL_FRAMES, 0)

    # Alternate, so a value that merely drifts with time cannot pass as
    # heading: it has to reverse when the wheel reverses.
    for _ in range(CYCLES):
        hold(STEER_LEFT, SEGMENT_FRAMES, -1)
        hold(STEER_RIGHT, SEGMENT_FRAMES, +1)

    return np.array(signs), np.array(speeds)


def wrapped_delta(values, modulus):
    """Change per sample, taking the shorter way round a wrapping value."""
    delta = np.diff(values.astype(np.int64), axis=0)
    half = modulus // 2
    delta[delta > half] -= modulus
    delta[delta < -half] += modulus
    return delta


def score(deltas, signs):
    """How cleanly each address turns with the steering.

    Rewards agreeing in sign with the input while turning, and punishes
    drifting while the wheel is straight.
    """
    turning = signs != 0
    if turning.sum() < 20:
        return None

    turn_delta = deltas[turning]
    turn_sign = signs[turning][:, None]

    agreement = np.sign(turn_delta) == turn_sign
    agree_fraction = agreement.mean(axis=0)

    # Movement while turning should dwarf movement while straight.
    moving = np.abs(turn_delta).mean(axis=0)
    straight = deltas[~turning]
    idle = np.abs(straight).mean(axis=0) if straight.size else np.zeros_like(moving)

    return agree_fraction, moving, idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=int, default=9)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--keep", action="store_true", help="leave the emulator running")
    args = parser.parse_args()

    dump_path = os.path.join(BASE_DIR, f"ram_heading{args.instance}.bin")
    for stale in (dump_path,):
        try:
            os.remove(stale)
        except OSError:
            pass

    print(f"launching emulator {args.instance} with RAM dumping...")
    process = launch(args.instance, dump_path)
    time.sleep(14)

    bridge = BridgeClient(instance=args.instance)
    print("driving the steering pattern...")
    signs, speeds = record(bridge)
    bridge.close()
    print(f"  speed while driving: min {speeds.min()}, "
          f"mean {speeds.mean():.0f}, max {speeds.max()}")

    time.sleep(1.0)
    if not args.keep:
        process.terminate()
        time.sleep(2)

    raw = np.fromfile(dump_path, dtype=np.uint8)
    samples = raw.size // WRAM_SIZE
    print(f"{samples} RAM samples, {len(signs)} steering labels")

    usable = min(samples, len(signs))
    if usable < 40:
        raise SystemExit("not enough samples -- did the emulator start?")

    ram = raw[: usable * WRAM_SIZE].reshape(usable, WRAM_SIZE)
    signs = signs[:usable]
    speeds = speeds[:usable]

    # A wrecked machine does not respond to the wheel, so those samples are
    # noise pretending to be labelled data.
    alive = speeds > 200
    print(f"  {alive.sum()}/{usable} samples with the machine actually moving")
    signs = np.where(alive, signs, 0)

    for label, values, modulus, stride in (
        ("8-bit", ram, 256, 1),
        ("16-bit", (ram[:, 0::2].astype(np.uint16)
                    | (ram[:, 1::2].astype(np.uint16) << 8)), 65536, 2),
    ):
        deltas = wrapped_delta(values, modulus)
        result = score(deltas, signs[1:])
        if result is None:
            continue

        agree, moving, idle = result

        # A heading turns with the wheel, moves meaningfully while turning,
        # and settles when the wheel is straight.
        candidates = np.flatnonzero((agree > 0.85) & (moving > 1.0) & (moving > idle * 3))
        ranked = sorted(candidates, key=lambda i: -agree[i] * moving[i])

        print()
        print(f"=== {label} heading candidates ===")
        if not ranked:
            print("  none")
        for index in ranked[: args.top]:
            print(
                f"  0x{index * stride:05X}  turns with the wheel "
                f"{agree[index] * 100:5.1f}% of the time, "
                f"{moving[index]:7.1f} per sample turning "
                f"vs {idle[index]:6.1f} straight"
            )

    print()
    print("Confirm the top candidate in BizHawk (Tools > RAM Watch): it should")
    print("sweep smoothly as you turn and wrap around rather than clamping.")


if __name__ == "__main__":
    main()
