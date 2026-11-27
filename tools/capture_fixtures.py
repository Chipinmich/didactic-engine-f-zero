"""Capture labelled frames for the race-view detector to be tested against.

The env decides an episode has ended by looking at the frame, and it gets
that wrong: the intro screen and the "YOU LOST" results screen are the same
low-saturation grey as the road, so they read as a race view and the agent
carries on "driving" a menu. Fixing that needs examples of both to test
against, and the only source of a real results screen is the game itself.

Two captures, neither of which can be labelled automatically -- that would
need the detector this is meant to test:

    python tools/capture_fixtures.py boot    frames from a cold ROM: title,
                                             menus, and the attract-mode demo

    python tools/capture_fixtures.py death   reset to the start state, then
                                             hold full throttle and full left
                                             lock until the machine is
                                             destroyed against a wall

Both write to vision/fixtures/unsorted/ with the state in the filename, e.g.
death_f000420_spd00000_cp07.png. Look at them, then sort them by hand into
vision/fixtures/racing/ and vision/fixtures/menu/. tests/test_race_view.py
reads those two directories.

Start an emulator first:

    python tools/launch_instances.py 1
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (
    ACTION_NONE,
    BASE_DIR,
    BridgeClient,
    BridgeTimeout,
    STEER_LEFT,
    SHOULDER_NONE,
    THROTTLE_ACCEL,
    encode_action,
)

UNSORTED_DIR = os.path.join(BASE_DIR, "vision", "fixtures", "unsorted")

# The boot sequence is long and mostly static, so sample it sparsely and for
# a long time -- the attract-mode demo race does not begin for a while.
BOOT_FRAMES = 5400
BOOT_EVERY = 30

# The death capture is the opposite: the transition from race to results is
# the whole point, so sample it densely.
#
# Full throttle and STRAIGHT does not work -- measured, it drove to
# checkpoint 50 still carrying 700 speed and never died, because the start
# state points down the road. Full left lock pins it against the wall, which
# is what actually drains power.
DEATH_FRAMES = 5400
DEATH_EVERY = 10


def save(frame, tag, index, speed, progress):
    if frame is None:
        return False

    name = f"{tag}_f{index:06d}_spd{speed:05d}_cp{progress:02d}.png"
    cv2.imwrite(os.path.join(UNSORTED_DIR, name), frame)
    return True


def capture(bridge, tag, action, total_frames, every, reset_first):
    if reset_first:
        print("Resetting to the start state...")
        bridge.reset()

    saved = 0
    for index in range(total_frames):
        try:
            speed, frame = bridge.advance(action)
        except BridgeTimeout as exc:
            print(f"\nbridge stopped responding at frame {index}: {exc}")
            break

        if index % every == 0:
            if save(frame, tag, index, speed, bridge.progress):
                saved += 1
            print(
                f"\r  frame {index:5d}/{total_frames}  speed {speed:5d}  "
                f"lap {bridge.lap}  cp {bridge.progress:2d}  saved {saved}",
                end="",
                flush=True,
            )

    print()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("boot", "death"))
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(UNSORTED_DIR, exist_ok=True)
    bridge = BridgeClient(instance=args.instance)

    try:
        if args.mode == "boot":
            # No input at all: the ROM plays through its own title sequence
            # and into the attract-mode demo.
            saved = capture(
                bridge,
                "boot",
                ACTION_NONE,
                args.frames or BOOT_FRAMES,
                BOOT_EVERY,
                reset_first=False,
            )
        else:
            # Full throttle into full left lock: the machine turns into the
            # wall and grinds along it, which drains power until it is
            # destroyed and the results screen appears.
            saved = capture(
                bridge,
                "death",
                encode_action(STEER_LEFT, THROTTLE_ACCEL, SHOULDER_NONE),
                args.frames or DEATH_FRAMES,
                DEATH_EVERY,
                reset_first=True,
            )
    finally:
        bridge.close()

    print(f"\n{saved} frames in {UNSORTED_DIR}")
    print("Sort them by hand into vision/fixtures/racing/ and vision/fixtures/menu/.")


if __name__ == "__main__":
    main()
