"""Find the RAM byte that says whether a race is actually in progress.

The env decides an episode is over by looking at the picture, and the
picture lies. The "1. MUTE CITY I" course card is the same flat road-grey as
the track, so it reads as a race view with a wide open road ahead; the
SELECT / TRY AGAIN menu and the "YOU LOST" screen fool it in their own ways.
The agent then feeds actions into a menu and trains on the result.

Pixel thresholds do not separate these: measured over 581 captured frames,
the "YOU LOST" screen sits at HUD-band contrast 65.9, right in the middle of
the racing distribution. So this looks for the game's own answer instead --
a byte that holds one value while racing and another on the menus, the same
way lap and progress were found (tools/find_ram.py).

Two passes:

    python tools/find_race_state.py capture     drive, die, repeat, while
                                                bridge.lua dumps work RAM

    python tools/find_race_state.py analyse     rank bytes by how cleanly
                                                they split racing from menu

The emulator must be started with the dump enabled, which launch_instances.py
does not do by default:

    python tools/launch_instances.py --stop
    FZERO_RAM_DUMP=ram_race.bin FZERO_RAM_EVERY=1 python tools/launch_instances.py 1
    python tools/find_race_state.py capture
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (
    ACTION_NONE,
    BASE_DIR,
    BridgeClient,
    BridgeTimeout,
    SHOULDER_NONE,
    STEER_LEFT,
    THROTTLE_ACCEL,
    encode_action,
)

WRAM_SIZE = 0x20000

DUMP = os.path.join(BASE_DIR, "ram_race.bin")
LOG = os.path.join(BASE_DIR, "ram_race.csv")
BOOT_LOG = os.path.join(BASE_DIR, "ram_boot.csv")
SHOTS = os.path.join(BASE_DIR, "vision", "fixtures", "race_state")

CAPTURE_FRAMES = 3000
SHOT_EVERY = 25

# Full throttle into full left lock: the machine grinds the wall, power
# drains, and it is destroyed. Measured -- going straight does not work, it
# simply drives the course.
KILL_ACTION = encode_action(STEER_LEFT, THROTTLE_ACCEL, SHOULDER_NONE)

# A frame is only labelled once it is unambiguous, because a wrong label
# poisons the search far worse than a missing one does.
#
# Menu: zero speed AND zero progress, held this long. Measured over six
# death cycles, the menu runs 39-45 frames every time, with single-frame
# zeros scattered through the racing stretches -- so 20 clears the noise
# without coming close to the real thing.
#
# Both conditions are needed: a machine can read zero speed for a frame
# mid-race, and progress zero alone is just the start line.
MENU_ZERO_RUN = 20

# Racing: moving properly, not the near-stationary moments around a death.
RACING_MIN_SPEED = 600


def capture(instance, total_frames, boot=False):
    if not os.path.exists(DUMP):
        raise SystemExit(
            f"No RAM dump at {DUMP}.\n"
            "Start the emulator with FZERO_RAM_DUMP set -- see the module "
            "docstring."
        )

    os.makedirs(SHOTS, exist_ok=True)
    bridge = BridgeClient(instance=instance)
    rows = []

    # Boot mode never resets, so the ROM plays its own title sequence and
    # attract demo. That is the other half of the bug -- the intro screen is
    # reported as a race view too -- and the death capture never visits it,
    # because it starts from a mid-race savestate.
    action = ACTION_NONE if boot else KILL_ACTION

    try:
        if not boot:
            bridge.reset()
        for i in range(total_frames):
            try:
                speed, frame = bridge.advance(action)
            except BridgeTimeout as exc:
                print(f"\nbridge stopped responding at {i}: {exc}")
                break

            # frame_count is the bridge's own counter, and with
            # FZERO_RAM_EVERY=1 it is exactly the RAM sample index. Taking
            # our own loop counter instead would drift the moment a frame is
            # dropped or the bridge advances without us.
            frame_count = bridge._frame_count

            rows.append(
                {
                    "sample": frame_count,
                    "speed": speed,
                    "lap": bridge.lap,
                    "progress": bridge.progress,
                }
            )

            if i % SHOT_EVERY == 0:
                if frame is not None:
                    cv2.imwrite(
                        os.path.join(SHOTS, f"s{frame_count:06d}.png"), frame
                    )
                print(
                    f"\r  {i:5d}/{total_frames}  sample {frame_count:6d}  "
                    f"speed {speed:5d}  cp {bridge.progress:2d}",
                    end="",
                    flush=True,
                )
    finally:
        bridge.close()

    log_path = BOOT_LOG if boot else LOG
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample", "speed", "lap", "progress"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} frames logged to {LOG}")
    print("Stop the emulator, then run: python tools/find_race_state.py analyse")


def label(log):
    """Split the log into confidently-racing and confidently-menu samples.

    Returns two index arrays into the RAM capture. Anything uncertain --
    the moments either side of a death, a machine merely slowed -- is left
    out of both.
    """
    speed = log["speed"]

    stopped = (speed == 0) & (log["progress"] == 0)

    # Run length ending at each frame, then spread the verdict back over the
    # whole run -- otherwise only the tail end of each menu is labelled and
    # the first 20 frames of it get thrown away.
    run = np.zeros(len(stopped), dtype=np.int32)
    for i, is_stopped in enumerate(stopped):
        run[i] = (run[i - 1] + 1) if (is_stopped and i > 0) else (1 if is_stopped else 0)

    menu = np.zeros(len(stopped), dtype=bool)
    for i in np.flatnonzero(run >= MENU_ZERO_RUN):
        menu[i - run[i] + 1 : i + 1] = True

    racing = speed >= RACING_MIN_SPEED

    return log["sample"][racing], log["sample"][menu]


def analyse():
    if not os.path.exists(LOG):
        raise SystemExit(f"No log at {LOG} -- run the capture pass first.")

    log = np.genfromtxt(LOG, delimiter=",", names=True, dtype=int)
    log = np.atleast_1d(log)

    raw = np.fromfile(DUMP, dtype=np.uint8)
    samples = raw.size // WRAM_SIZE
    ram = raw[: samples * WRAM_SIZE].reshape(samples, WRAM_SIZE)
    print(f"{samples} RAM samples, {len(log)} logged frames")

    racing_idx, menu_idx = label(log)
    racing_idx = racing_idx[racing_idx < samples]
    menu_idx = menu_idx[menu_idx < samples]

    print(f"confidently racing: {len(racing_idx)} samples")
    print(f"confidently menu  : {len(menu_idx)} samples")

    if len(racing_idx) < 20 or len(menu_idx) < 20:
        raise SystemExit(
            "Not enough of both classes. The machine has to actually die -- "
            "check vision/fixtures/race_state/ for a results screen."
        )

    racing = ram[racing_idx]
    menu = ram[menu_idx]

    # A mode flag holds one value throughout a race and a different one on
    # the menus. Constant within each class is the whole test; anything that
    # wobbles is physics state, not a mode.
    racing_same = (racing == racing[0]).all(axis=0)
    menu_same = (menu == menu[0]).all(axis=0)
    differs = racing[0] != menu[0]

    hits = np.flatnonzero(racing_same & menu_same & differs)

    print(f"\n{len(hits)} addresses hold one value while racing and another on the menus\n")
    print(f"{'address':>10}  {'racing':>7}  {'menu':>6}")
    for addr in hits[:40]:
        print(f"  0x{addr:05X}  {racing[0][addr]:7d}  {menu[0][addr]:6d}")

    if len(hits) > 40:
        print(f"  ... and {len(hits) - 40} more")

    print(
        "\nConfirm a candidate in BizHawk before trusting it: watch it while "
        "driving, dying, and sitting on the menu."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("capture", "analyse"))
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--frames", type=int, default=CAPTURE_FRAMES)
    parser.add_argument("--boot", action="store_true",
                        help="no reset: capture the title sequence and attract demo")
    args = parser.parse_args()

    if args.mode == "capture":
        capture(args.instance, args.frames, boot=args.boot)
    else:
        analyse()


if __name__ == "__main__":
    main()
