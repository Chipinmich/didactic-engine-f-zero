"""Standalone seven-ray F-Zero autopilot.

This controller is deliberately separate from PPO, DQN and neuroevolution.
It is a measurable reference driver, not an automatic teacher.  Acceleration
is always held; the only decisions are steering and same-direction lean.

The seven forward rays provide the desired open direction.  Near road rows
pull the machine back toward the centre, and shrinking ray distances add an
anticipatory correction before the wall reaches the car.  Since SNES input
is digital, proportional steering is delivered as short committed bursts.

Usage (only when training emulators are stopped):
    python ray_autopilot.py --reset --max-seconds 60
"""

import argparse
import csv
import time

import numpy as np

from bridge_client import (
    SHOULDER_L,
    SHOULDER_NONE,
    SHOULDER_R,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    BridgeClient,
    BridgeTimeout,
    encode_action,
)
from training.session_lock import EmulatorSessionLock
from vision.road_edges import (
    SCAN_ROWS,
    measure_rays,
    measure_rows,
    touching_wall,
)


# Targets are acceptance criteria, not reward.  Distance and speed must also
# be checked before this controller is ever considered a useful teacher.
TARGET_CENTERING = 0.85
TARGET_MAX_WALL_CONTACT = 0.02
TARGET_RAY_RESPONSE = 0.55
TARGET_FORWARD_CLEARANCE = 0.65

STEER_DEADZONE = 0.05
STEER_FULL_SCALE = 0.42
STEER_SMOOTHING = 0.40
RAY_SMOOTHING = 0.45
CENTER_GAIN = 0.65
RAY_GAIN = 1.10
CLOSING_GAIN = 1.25
HAZARD_GAIN = 1.20
CRITICAL_CLEARANCE = 0.18
LEAN_THRESHOLD = 0.28
REVERSAL_HOLD_FRAMES = 6
MIN_TURN_BURST = 2
MAX_TURN_BURST = 6
TARGET_SIDE_CLEARANCE_PX = 55
RAY_RESPONSE_THRESHOLD = 0.10


def _side_offset(reading, target_clearance=TARGET_SIDE_CLEARANCE_PX):
    """Signed centering correction: negative left, positive right."""
    if not reading.get("on_road", True):
        nearest = reading.get("nearest_road_x")
        centre = reading.get("center_x", 128)
        if nearest is None or nearest == centre:
            return 0.0
        return 1.0 if nearest > centre else -1.0

    left = reading.get("left_px")
    right = reading.get("right_px")
    if left is not None and right is not None:
        width = left + right
        return 0.0 if width <= 0 else float(np.clip((right - left) / width, -1, 1))
    if left is not None and left < target_clearance:
        return (target_clearance - left) / target_clearance
    if right is not None and right < target_clearance:
        return -(target_clearance - right) / target_clearance
    return 0.0


def centering_quality(readings):
    """1.0 at the road centre and 0.0 at/off an edge."""
    if not readings:
        return 0.0
    near = readings[-1]
    if not near.get("on_road", True):
        return 0.0
    left, right = near.get("left_px"), near.get("right_px")
    if left is None or right is None or left + right <= 0:
        return 1.0
    return 1.0 - min(abs(right - left) / (left + right), 1.0)


class SevenRayAutopilot:
    """Map-independent proportional steering and lean from road geometry."""

    def __init__(self):
        self._smooth_rays = None
        self._previous_rays = None
        self._smooth_signal = 0.0
        self._duty = 0.0
        self._pulse_remaining = 0
        self._last_direction = STEER_STRAIGHT
        self._direction_age = 0
        self._recovery_direction = None
        self.last_signal = 0.0
        self.last_ray_signal = 0.0
        self.last_center_signal = 0.0
        self.last_forward_clearance = 0.0

    @staticmethod
    def _seven(values):
        rays = np.zeros(7, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).ravel()
        rays[: min(values.size, 7)] = values[:7]
        return np.clip(rays, 0.0, 1.0)

    def _predict_rays(self, values):
        current = self._seven(values)
        if self._smooth_rays is None:
            smooth = current.copy()
            closing = np.zeros_like(current)
        else:
            smooth = (
                RAY_SMOOTHING * current
                + (1.0 - RAY_SMOOTHING) * self._smooth_rays
            )
            closing = np.maximum(self._previous_rays - current, 0.0)
        self._previous_rays = current
        self._smooth_rays = smooth
        return np.clip(smooth - CLOSING_GAIN * closing, 0.0, 1.0)

    @staticmethod
    def _open_direction(rays):
        # Inner diagonal rays predict ordinary corners; outer rays are useful
        # for emergencies but are less stable under perspective distortion.
        positions = np.array([-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0])
        weights = np.array([0.35, 0.70, 1.0, 0.20, 1.0, 0.70, 0.35])
        scale = float(np.sum(np.abs(positions) * weights))
        return float(np.sum(positions * weights * rays) / max(scale, 1e-6))

    @staticmethod
    def _hazard_direction(rays):
        left = float(np.min(rays[:3]))
        right = float(np.min(rays[4:]))
        if min(left, right) >= CRITICAL_CLEARANCE:
            return 0.0
        if abs(left - right) < 0.02:
            left = float(np.mean(rays[:3]))
            right = float(np.mean(rays[4:]))
        danger = max((CRITICAL_CLEARANCE - min(left, right)) / CRITICAL_CLEARANCE, 0.0)
        return danger if right > left else -danger

    def _recovery(self, readings):
        if readings and readings[-1].get("on_road", True):
            self._recovery_direction = None
            return None
        if not readings:
            return None
        if self._recovery_direction is None:
            vote = sum(np.sign(_side_offset(reading)) for reading in readings)
            if vote:
                self._recovery_direction = 1.0 if vote > 0 else -1.0
        return self._recovery_direction

    def _centering_signal(self, readings):
        if not readings:
            return 0.0
        # SCAN_ROWS is far-to-near.  The nearest row dominates positioning;
        # the next row supplies a small amount of lookahead stability.
        near = _side_offset(readings[-1])
        middle = _side_offset(readings[-2]) if len(readings) > 1 else near
        return 0.80 * near + 0.20 * middle

    def _steer(self, raw_signal, emergency=False):
        self._smooth_signal += STEER_SMOOTHING * (raw_signal - self._smooth_signal)
        signal = self._smooth_signal
        magnitude = abs(signal)
        wanted = STEER_RIGHT if signal > 0 else STEER_LEFT

        if emergency:
            self._pulse_remaining = 0
            self._duty = 0.0
            self._last_direction = wanted
            self._direction_age += 1
            return wanted

        if magnitude < STEER_DEADZONE:
            self._pulse_remaining = 0
            self._duty *= 0.5
            self._direction_age = 0
            return STEER_STRAIGHT

        if (
            self._last_direction not in (STEER_STRAIGHT, wanted)
            and self._direction_age < REVERSAL_HOLD_FRAMES
            and magnitude < STEER_FULL_SCALE
        ):
            self._direction_age += 1
            self._pulse_remaining = 0
            return STEER_STRAIGHT

        demand = min(
            (magnitude - STEER_DEADZONE)
            / max(STEER_FULL_SCALE - STEER_DEADZONE, 1e-6),
            1.0,
        )
        self._duty += demand

        if self._pulse_remaining > 0 and wanted == self._last_direction:
            self._pulse_remaining -= 1
            self._duty -= 1.0
            self._direction_age += 1
            return wanted

        if self._duty < 1.0:
            return STEER_STRAIGHT

        burst = MIN_TURN_BURST + int(
            round((MAX_TURN_BURST - MIN_TURN_BURST) * demand)
        )
        self._duty -= 1.0
        self._pulse_remaining = burst - 1
        self._direction_age = 1 if wanted != self._last_direction else self._direction_age + 1
        self._last_direction = wanted
        return wanted

    def update(self, readings, ray_values, speed=None):
        """Return ``((steer, accelerate, shoulder), explanation)``."""
        predicted = self._predict_rays(ray_values)
        ray_signal = self._open_direction(predicted)
        hazard_signal = self._hazard_direction(predicted)
        center_signal = self._centering_signal(readings)
        recovery = self._recovery(readings)

        self.last_ray_signal = ray_signal
        self.last_center_signal = center_signal
        self.last_forward_clearance = float(np.mean(predicted))

        if recovery is not None:
            signal = recovery
            emergency = True
            reason = "recover road"
        else:
            signal = float(np.clip(
                CENTER_GAIN * center_signal
                + RAY_GAIN * ray_signal
                + HAZARD_GAIN * hazard_signal,
                -1.0,
                1.0,
            ))
            emergency = abs(hazard_signal) > 0.0
            reason = "avoid wall" if emergency else "follow opening"

        self.last_signal = signal
        steer = self._steer(signal, emergency=emergency)
        if abs(self._smooth_signal) >= LEAN_THRESHOLD or emergency:
            shoulder = SHOULDER_R if signal > 0 else SHOULDER_L
        else:
            shoulder = SHOULDER_NONE

        direction = {STEER_LEFT: "left", STEER_RIGHT: "right"}.get(steer, "straight")
        return (
            (steer, THROTTLE_ACCEL, shoulder),
            f"{reason}: {direction} demand={signal:+.2f} "
            f"clear={self.last_forward_clearance:.0%}",
        )


class AutopilotMetrics:
    def __init__(self):
        self.frames = 0
        self.centering_total = 0.0
        self.wall_frames = 0
        self.ray_response_opportunities = 0
        self.ray_responses = 0
        self.clearance_total = 0.0

    def update(self, readings, ray_values, steer, ray_signal):
        self.frames += 1
        self.centering_total += centering_quality(readings)
        self.wall_frames += int(touching_wall(readings))
        rays = SevenRayAutopilot._seven(ray_values)
        self.clearance_total += float(np.mean(rays))
        if abs(ray_signal) >= RAY_RESPONSE_THRESHOLD:
            self.ray_response_opportunities += 1
            wanted = STEER_RIGHT if ray_signal > 0 else STEER_LEFT
            self.ray_responses += int(steer == wanted)

    def summary(self):
        frames = max(self.frames, 1)
        opportunities = max(self.ray_response_opportunities, 1)
        return {
            "centering": self.centering_total / frames,
            "wall_contact": self.wall_frames / frames,
            "ray_response": self.ray_responses / opportunities,
            "forward_clearance": self.clearance_total / frames,
        }


def _format_metrics(metrics):
    return "  ".join(
        (
            f"centre {metrics['centering']:.0%}/{TARGET_CENTERING:.0%}",
            f"wall {metrics['wall_contact']:.1%}/{TARGET_MAX_WALL_CONTACT:.0%} max",
            f"ray response {metrics['ray_response']:.0%}/{TARGET_RAY_RESPONSE:.0%}",
            f"clearance {metrics['forward_clearance']:.0%}/{TARGET_FORWARD_CLEARANCE:.0%}",
        )
    )


def run(args):
    controller = SevenRayAutopilot()
    metrics = AutopilotMetrics()
    bridge = BridgeClient(instance=args.instance)
    action = encode_action(STEER_STRAIGHT, THROTTLE_ACCEL, SHOULDER_NONE)
    started = time.time()
    last_print = 0.0
    non_racing = 0
    writer = None
    handle = None

    if args.log:
        handle = open(args.log, "w", newline="", encoding="utf-8")
        fields = [
            "frame", "elapsed", "lap", "progress", "speed", "steer",
            "shoulder", "centering", "wall_contact", "ray_response",
            "forward_clearance", *[f"ray_{index}" for index in range(7)],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

    print("Seven-ray autopilot: acceleration always held; Ctrl+C stops safely.")
    print("Targets:", _format_metrics(AutopilotMetrics().summary()))
    if args.reset:
        bridge.reset(slot=args.reset_slot)

    try:
        frame_number = 0
        while not args.max_seconds or time.time() - started < args.max_seconds:
            speed, frame = bridge.advance(action)
            frame_number += 1
            if frame is None:
                continue

            readings = measure_rows(frame, SCAN_ROWS)
            ray_values = [ray["distance"] for ray in measure_rays(frame)]
            dims, reason = controller.update(readings, ray_values, speed)
            action = encode_action(*dims)
            metrics.update(readings, ray_values, dims[0], controller.last_ray_signal)

            non_racing = 0 if bridge.racing else non_racing + 1
            if non_racing >= 30:
                print("Race is no longer active; stopping autopilot.")
                break

            current = metrics.summary()
            if writer:
                row = {
                    "frame": frame_number,
                    "elapsed": time.time() - started,
                    "lap": bridge.lap,
                    "progress": bridge.progress,
                    "speed": speed,
                    "steer": dims[0],
                    "shoulder": dims[2],
                    **current,
                }
                row.update({f"ray_{i}": value for i, value in enumerate(ray_values[:7])})
                writer.writerow(row)

            if time.time() - last_print >= args.print_interval:
                print(
                    f"lap {bridge.lap} cp {bridge.progress:02d} speed {speed:4d}  "
                    f"{reason}  {_format_metrics(current)}"
                )
                last_print = time.time()
    except (BridgeTimeout, KeyboardInterrupt) as exc:
        if isinstance(exc, BridgeTimeout):
            print(f"[error] {exc}")
    finally:
        bridge.close()
        if handle:
            handle.close()
        print("Final:", _format_metrics(metrics.summary()))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--reset-slot", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--log", metavar="CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    # A standalone/default bridge owns the whole emulator space.  An explicit
    # instance owns only that index.  Either form refuses overlap with live
    # GA/DQN locks instead of disrupting their action files.
    count = None if args.instance is None else 1
    offset = 0 if args.instance is None else args.instance
    with EmulatorSessionLock("seven-ray autopilot", offset=offset, count=count):
        run(args)


if __name__ == "__main__":
    main()
