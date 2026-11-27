"""Build a savestate the RL can start every episode from.

Resetting to the hand-saved start line means every episode begins
stationary on the grid -- a position the agent barely revisits, so the first
thing it must learn is how to get out of it. Warming up at the start of
each episode does not fix that: driving up to speed depends on driving
well, which is the thing that does not work yet. Measured, it produced
426 +- 115 rather than the speed asked for.

So the state is built ONCE and reused. Two passes, because a savestate has
to be written AT the chosen moment and the best moment is not known until
the drive is over:

  pass 1  drive and record speed, centring and power every frame
  pass 2  drive again and save the slot at the best frame

Both the emulator and the controller are deterministic, so pass 2 retraces
pass 1 exactly.

    python tools/make_start_state.py

Then launch the emulators with FZERO_START_SLOT=2.
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
    SHOULDER_NONE,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    BridgeClient,
    encode_action,
)
from control import SCAN_FRACTIONS, Controller
from vision.hud import read_power
from vision.road_edges import measure_rows

BIZHAWK_DIR = r"C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
EMUHAWK = os.path.join(BIZHAWK_DIR, "EmuHawk.exe")
ROM = os.path.join(BIZHAWK_DIR, "F-Zero (USA).sfc")
LUA = os.path.join(BASE_DIR, "lua", "bridge.lua")

SAVE_ACTION = 25

# Centring is measured well up the screen, NOT on the row the machine sits
# on. Perspective makes the road fill the view close up, so the near row
# shows both walls on only about 20% of frames against 90% at row 70 --
# measuring there made centring unreadable on 97% of frames.
CENTRE_ROW = 70

CENTRE_TOLERANCE = 0.16
MIN_POWER = 0.90
MIN_SPEED = 600

DRIVE_FRAMES = 900


def launch(instance, save_slot):
    env = dict(
        os.environ,
        FZERO_INSTANCE=str(instance),
        FZERO_SAVE_SLOT=str(save_slot),
        FZERO_START_SLOT="1",
    )
    return subprocess.Popen([EMUHAWK, ROM, f"--lua={LUA}"], env=env, cwd=BIZHAWK_DIR)


def drive(bridge, save_at=None):
    """Drive from a reset. Saves the slot at frame `save_at` if given.

    Returns per-frame speed, centring error and power.
    """
    controller = Controller()
    bridge.reset()

    speeds, offsets, powers = [], [], []
    frame = None
    speed = 0

    for index in range(DRIVE_FRAMES):
        if save_at is not None and index == save_at:
            # Edge-triggered in the bridge, so send it twice to be sure the
            # change of action is seen.
            bridge.advance(SAVE_ACTION)
            bridge.advance(SAVE_ACTION)
            return speeds, offsets, powers

        if frame is None:
            action = encode_action(STEER_STRAIGHT, THROTTLE_ACCEL, SHOULDER_NONE)
        else:
            height, width = frame.shape[:2]
            readings = measure_rows(
                frame, [int(height * f) for f in SCAN_FRACTIONS]
            )
            steer, _, shoulder = controller.update(readings, width / 2.0, speed)[0]
            action = encode_action(steer, THROTTLE_ACCEL, shoulder)

        speed, frame = bridge.advance(action)

        if frame is None:
            speeds.append(0)
            offsets.append(np.nan)
            powers.append(np.nan)
            continue

        near = measure_rows(frame, [CENTRE_ROW])[0]
        left, right = near["left_px"], near["right_px"]
        if left is None or right is None or (left + right) <= 0:
            offset = np.nan
        else:
            offset = abs(right - left) / (left + right)

        power = read_power(frame)

        speeds.append(speed)
        offsets.append(offset)
        powers.append(power if power is not None else np.nan)

    return speeds, offsets, powers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=int, default=9)
    parser.add_argument("--slot", type=int, default=2)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    print(f"launching emulator {args.instance}...")
    process = launch(args.instance, args.slot)
    time.sleep(14)

    bridge = BridgeClient(instance=args.instance)

    print("pass 1: driving and recording...")
    speeds, offsets, powers = drive(bridge)

    speeds = np.array(speeds, dtype=float)
    offsets = np.array(offsets, dtype=float)
    powers = np.array(powers, dtype=float)

    qualifies = (
        (offsets <= CENTRE_TOLERANCE)
        & (powers >= MIN_POWER)
        & (speeds >= MIN_SPEED)
    )
    candidates = np.flatnonzero(qualifies)

    print(f"  {len(speeds)} frames, {candidates.size} qualify "
          f"(centred <= {CENTRE_TOLERANCE:.0%}, power >= {MIN_POWER:.0%}, "
          f"speed >= {MIN_SPEED})")

    if candidates.size == 0:
        print(f"  best speed {np.nanmax(speeds):.0f}, "
              f"best centring {np.nanmin(offsets) * 100:.0f}%")
        print("\n! no frame met all three. Lower MIN_SPEED or widen the")
        print("  tolerance, or save a state by hand in BizHawk instead.")
        bridge.close()
        if not args.keep:
            process.terminate()
        return 1

    # Of the qualifying moments, take the fastest.
    best = int(candidates[np.argmax(speeds[candidates])])
    print(f"  best frame {best}: speed {speeds[best]:.0f}, "
          f"off-centre {offsets[best] * 100:.0f}%, power {powers[best]:.0%}")

    print("pass 2: replaying and saving...")
    drive(bridge, save_at=best)

    bridge.read_full_state()
    saves = bridge.save_count
    bridge.close()

    time.sleep(1)
    if not args.keep:
        process.terminate()
        time.sleep(2)

    print()
    print(f"saved to slot {args.slot} (bridge reports {saves} save(s))")
    print(f"Launch emulators with FZERO_START_SLOT={args.slot} to start there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
