"""Work out empirically what each SNES button does in F-Zero.

bridge.lua assumes B accelerates and Y brakes. Rather than trusting that,
this drives the car up to speed and then holds each candidate button,
measuring what happens to the speed value.

Run it with the race started and the car able to move freely:

    python tools/find_buttons.py

Interpretation:
  - a button that raises speed from a standstill is the accelerator
  - a button that drops speed sharply while coasting is the brake
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bridge_client import (
    ACTION_NONE,
    RAW_BUTTONS,
    BridgeClient,
    BridgeTimeout,
    raw_button_action,
)

# Short windows on purpose. Accelerating in a straight line from the
# savestate reaches a wall in well under two seconds, and a crash drops
# speed to zero -- which reads exactly like a very effective brake. Every
# reading below has to happen before the car can hit anything.
SPOOL_UP_FRAMES = 70
TEST_FRAMES = 12

# Coasting sheds ~80 units a frame; anything far beyond that in a single
# frame is an impact, not deceleration.
CRASH_DROP_PER_FRAME = 400

def hold(bridge, action, frames):
    speed = 0
    for _ in range(frames):
        speed, _ = bridge.advance(action)
    return speed


def hold_trace(bridge, action, frames):
    """Speed at every frame, so a crash is distinguishable from braking."""
    trace = []
    for _ in range(frames):
        speed, _ = bridge.advance(action)
        trace.append(speed)
    return trace


def looks_like_crash(trace, start):
    """A crash collapses in a frame or two; coasting decays linearly.

    Reaching zero is NOT the test -- F-Zero sheds about 80 units a frame
    when coasting, so it hits zero on its own in well under a second.
    """
    previous = start
    for value in trace:
        if previous - value > CRASH_DROP_PER_FRAME:
            return True
        previous = value
    return False


def decel_rate(trace, start):
    """Mean speed lost per frame, over the part before it bottoms out.

    Comparing final speeds is useless once coasting has already reached
    zero: nothing can score lower, so every button ties.
    """
    values = [start] + [v for v in trace]
    drops = []
    for previous, value in zip(values, values[1:]):
        if value <= 0:
            break
        drops.append(previous - value)
    return sum(drops) / len(drops) if drops else 0.0


def main():
    bridge = BridgeClient()

    print("Which button accelerates (held from a standstill)?")
    print()

    accelerators = []
    for name in RAW_BUTTONS:
        bridge.reset()
        start = hold(bridge, ACTION_NONE, 30)
        end = hold(bridge, raw_button_action(name), TEST_FRAMES)
        gained = end - start
        if gained > 50:
            accelerators.append((gained, name))
        print(f"  {name:2s}: {start:6d} -> {end:6d}  ({gained:+d})")

    print()
    if accelerators:
        best = max(accelerators)[1]
        print(f"  -> accelerator looks like: {best}")
    else:
        print("  ! Nothing accelerated. Is the car able to move, or is the")
        print("    race still on a countdown / results screen?")

    print()
    print("Which button brakes (held after spooling up)?")
    print()

    accel = raw_button_action("B")

    bridge.reset()
    spooled = hold(bridge, accel, SPOOL_UP_FRAMES)
    coast_trace = hold_trace(bridge, ACTION_NONE, TEST_FRAMES)
    coast_decel = decel_rate(coast_trace, spooled)
    print(f"  coast (baseline): from {spooled:5d}, losing "
          f"{coast_decel:5.0f}/frame")
    print(f"    trace: {coast_trace}")
    print()

    brakes = []
    for name in RAW_BUTTONS:
        bridge.reset()
        spooled = hold(bridge, accel, SPOOL_UP_FRAMES)
        trace = hold_trace(bridge, raw_button_action(name), TEST_FRAMES)
        rate = decel_rate(trace, spooled)

        if looks_like_crash(trace, spooled):
            note = "[impact - ignore]"
        elif rate > coast_decel * 1.15:
            note = f"SLOWS FASTER than coasting ({rate/max(coast_decel,1):.2f}x)"
            brakes.append((rate, name))
        elif rate < coast_decel * 0.5:
            note = "keeps speed up (accelerates)"
        else:
            note = "same as coasting"

        print(f"  {name:2s}: from {spooled:5d}, losing {rate:5.0f}/frame  {note}")
        print(f"    trace: {trace}")

    print()
    if brakes:
        best = max(brakes)[1]
        print(f"  -> brake looks like: {best}")
        print(f"     set brake_button = \"{best}\" in lua/bridge.lua")
    else:
        print("  -> NO DEDICATED BRAKE on this ROM.")
        print("     Nothing sheds speed faster than simply releasing the")
        print("     accelerator, which already loses ~%.0f/frame." % coast_decel)
        print("     Drop THROTTLE_BRAKE so throttle is just coast/accelerate;")
        print("     coasting IS the brake, and L/R are the air brakes.")

    bridge.close()


if __name__ == "__main__":
    try:
        main()
    except BridgeTimeout as exc:
        print(f"[error] {exc}")
