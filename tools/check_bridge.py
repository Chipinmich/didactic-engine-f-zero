"""Diagnostics for the EmuHawk <-> Python bridge.

Run this with the Lua script loaded and the race running. It advances the
emulator itself (the bridge is lockstep), and reports whether the
memory-mapped frame transport works, how fast frames arrive, what the road
detector sees, and the peak raw speed value so MAX_SPEED_RAW in
env/fzero_env.py can be calibrated.

    python tools/check_bridge.py
"""

import os
import subprocess
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bridge_client import (
    ACTION_ACCEL,
    MMF_NAME,
    PNG_SIGNATURE,
    STATE_FILE,
    BridgeClient,
    BridgeTimeout,
)
from vision.road_edges import SCAN_ROWS, extract_features, measure_edges

DURATION_SECONDS = 20.0

# state.txt is rewritten every emulated frame, so anything older than this
# means nothing is driving the bridge right now.
STALE_AFTER_SECONDS = 2.0

# Below this many advanced frames, any speed reading is a leftover rather
# than something the car actually did during this run.
MIN_STEPS_FOR_CALIBRATION = 10


def emuhawk_running():
    """True if EmuHawk is in the process list, None if we couldn't tell."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq EmuHawk.exe"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return "EmuHawk.exe" in out


def state_age_seconds():
    try:
        return time.time() - os.path.getmtime(STATE_FILE)
    except OSError:
        return None


def main():
    bridge = BridgeClient()

    print("=== emulator ===")
    running = emuhawk_running()
    if running is False:
        print("  ! EmuHawk is NOT running. Start it with the ROM and load")
        print("    lua/bridge.lua, or just use: run.bat check")
    elif running:
        print("  EmuHawk process found")
    else:
        print("  (could not check the process list)")

    print()
    print("=== state.txt ===")
    speed, resets, frames = bridge.read_state()
    age = state_age_seconds()

    print(f"speed={speed} resets={resets} frames={frames}")
    if age is None:
        print("  ! state.txt does not exist -- the bridge has never run.")
    elif age > STALE_AFTER_SECONDS:
        print(f"  ! state.txt is {age:.0f}s old. bridge.lua rewrites it every")
        print("    frame, so nothing is driving the bridge. Any numbers above")
        print("    are leftovers from an earlier run -- do not calibrate to them.")
    elif frames == -1:
        print("  ! No frame counter. bridge.lua looks out of date -- reload")
        print("    the script in EmuHawk (Lua Console -> Script -> Reload).")
    else:
        print(f"  fresh ({age:.1f}s old)")

    print()
    print("=== memory-mapped frame ===")
    mmf = bridge._open_mmf()
    if mmf is None:
        print(f"  ! Could not open mapping '{MMF_NAME}' at any size.")
        print("    Set USE_MMF = false in bridge.lua to fall back to disk.")
    else:
        mmf.seek(0)
        head = mmf.read(8)
        print(f"  mapping size {mmf.size()} bytes")
        print(f"  first bytes: {head.hex(' ')}")

        if head.startswith(PNG_SIGNATURE):
            print("  -> PNG signature at offset 0, framing is as expected")
        elif not any(head):
            # mmap(-1, ...) creates the mapping when none exists, so an empty
            # buffer means we made it ourselves rather than finding BizHawk's.
            print(f"  ! Mapping '{MMF_NAME}' is empty -- BizHawk has not written")
            print("    a frame. This almost always means the Lua script isn't")
            print("    running, or is an older version without comm.mmfScreenshot.")
        else:
            print("  ! Data present but not a PNG. BizHawk may frame this")
            print("    differently in your build; set USE_MMF = false to")
            print("    fall back to reading frames from disk.")

    frame = bridge.read_frame()
    print(f"  decoded frame: {None if frame is None else frame.shape}")

    print()
    print(f"=== live capture ({DURATION_SECONDS:.0f}s, accelerating) ===")

    max_speed = 0
    decoded = 0
    failed = 0
    steps = 0

    start = time.monotonic()
    last_print = start

    try:
        while time.monotonic() - start < DURATION_SECONDS:
            speed, frame = bridge.advance(ACTION_ACCEL)
            steps += 1
            max_speed = max(max_speed, speed)

            if frame is None:
                failed += 1
                continue
            decoded += 1

            now = time.monotonic()
            if now - last_print >= 2.0:
                last_print = now
                h = frame.shape[0]
                near = measure_edges(frame, scan_y=int(h * 0.82))
                print(
                    f"  speed={speed:6d}  left={near['left_px']}  "
                    f"right={near['right_px']}  "
                    f"features={extract_features(frame, 0.0)[:4].round(2)}"
                )
    except BridgeTimeout as exc:
        print(f"  ! {exc}")

    elapsed = time.monotonic() - start

    print()
    print("=== results ===")
    print(f"  steps advanced  : {steps}  ({steps / elapsed:.1f}/s)")
    print(f"  frames decoded  : {decoded}")
    print(f"  decode failures : {failed}")
    print(f"  peak raw speed  : {max_speed}")

    if steps < MIN_STEPS_FOR_CALIBRATION:
        print()
        print(f"  ! Only {steps} step(s) advanced, so nothing here is a real")
        print("    reading -- the speed above is whatever was left in")
        print("    state.txt from an earlier run. Do NOT set MAX_SPEED_RAW")
        print("    from it. Fix the bridge first, then re-run this.")
    elif max_speed > 0:
        print(f"  -> set MAX_SPEED_RAW to about {max_speed} in env/fzero_env.py")
    else:
        print("  ! Speed never rose above 0. Either the car never moved, or")
        print("    the RAM address 0x000B20 in bridge.lua is wrong.")

    bridge.close()


if __name__ == "__main__":
    main()
