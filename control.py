"""Steering controller for the F-Zero autopilot.

Replaces a bang-bang controller that used a single scan row and could only
react once a wall was already close. That could not finish a lap: by the
time a tight corner registered on the near row, the car was carrying too
much speed to make it.

Three things changed:

  lookahead    three scan rows, not one. The far rows show where the track
               goes next, so a corner is steered for before it arrives.

  proportional the steering input is binary per frame, but the loop runs at
               60fps, so a fractional steering amount is delivered by
               steering on a fraction of frames. A gentle drift gets a
               light correction instead of full lock.

  lean+brake   large corrections add the shoulder button, which is how a
               tight corner is actually taken at speed, and a sharp enough
               corner ahead lifts off or brakes before entry.
"""

from bridge_client import (
    COAST_DECEL_PER_FRAME,
    SHOULDER_L,
    SHOULDER_NONE,
    SHOULDER_R,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    THROTTLE_COAST,
    TOP_SPEED_RAW,
)

# Scan rows as a fraction of frame height: the row the car sits on, plus two
# lookahead rows further up the screen (further down the track).
SCAN_FRACTIONS = (0.82, 0.60, 0.45)

# How much each row contributes to the steering signal. The near row keeps
# the car centred; the far rows anticipate.
ROW_WEIGHTS = (1.0, 0.6, 0.35)

# Below this the car is considered centred enough to run straight, which
# stops it hunting left-right on a straight.
#
# These four values were picked by racing configurations against each other
# on parallel emulators (tools/compare_configs.py), scored on how far round
# the track they got and how much time they spent scraping a wall. Centring
# harder cut wall contact from 23% to 16% and got furthest; going harder
# still backfired badly (38% wall contact), so this is a middle setting,
# not a maximum.
STEER_DEADZONE = 0.04

# Signal magnitude that corresponds to full-lock steering every frame.
# Higher means less steering for the same error -- the car was reacting to
# every pixel of noise and sawing at the wheel.
STEER_FULL_SCALE = 0.35

# Above this, add shoulder lean to tighten the turn.
LEAN_THRESHOLD = 0.30

# Throttle is a closed loop on a target speed rather than a severity
# threshold, because coasting is far too blunt an instrument to use on a
# timer: it sheds 80 speed units a frame while the throttle only gains 19,
# so every frame off-throttle costs four to win back. Holding a fixed
# "coast while the corner looks sharp" rule bleeds enormous speed.
#
# Instead: estimate a speed the upcoming curvature can be taken at, coast
# only while above it, and get straight back on the throttle underneath.
# That self-corrects, so it stops coasting the instant it has slowed enough.
CURVE_SLOWDOWN = 1.9

# Never aim below this, or a mild bend crawls.
MIN_CORNER_SPEED = 800

# Curvature below this is a straight; aim for everything the car has.
STRAIGHT_SEVERITY = 0.12

# The road is dithered and there are only three scan rows, so the raw demand
# is noisy frame to frame and steering on it directly makes the car saw
# left-right. This is an exponential moving average: 1.0 is no smoothing,
# lower is steadier but slower to react.
STEER_SMOOTHING = 0.25

# Frames a steering direction is held before it may reverse, unless the
# demand the other way is at full scale (a real corner, not noise).
REVERSAL_HOLD_FRAMES = 8

# Steering demand used when the car has left the track entirely. Full lock
# toward the road: nothing else matters until it is back on.
OFF_ROAD_DEMAND = 1.0

# When only one wall is visible the road runs off-screen the other way, so
# there is no centre to steer to. React only if the visible wall is closer
# than this many pixels.
TARGET_CLEARANCE = 55



class Controller:
    def __init__(
        self,
        deadzone=STEER_DEADZONE,
        full_scale=STEER_FULL_SCALE,
        lean_threshold=LEAN_THRESHOLD,
        curve_slowdown=CURVE_SLOWDOWN,
        min_corner_speed=MIN_CORNER_SPEED,
        smoothing=STEER_SMOOTHING,
        reversal_hold=REVERSAL_HOLD_FRAMES,
        row_weights=ROW_WEIGHTS,
        target_clearance=TARGET_CLEARANCE,
        allow_lean=True,
    ):
        self.deadzone = deadzone
        self.full_scale = full_scale
        self.lean_threshold = lean_threshold
        self.curve_slowdown = curve_slowdown
        self.min_corner_speed = min_corner_speed
        self.smoothing = smoothing
        self.reversal_hold = reversal_hold
        self.row_weights = row_weights
        self.target_clearance = target_clearance
        self.allow_lean = allow_lean

        self.last_target = TOP_SPEED_RAW
        self.last_severity = 0.0

        # Carries the fractional part of the steering demand between frames.
        self._duty = 0.0

        # Smoothed steering demand, plus the direction currently held and
        # how long it has been held -- see _steer().
        self._smoothed = 0.0
        self._last_direction = None
        self._hold = 0

        # Which way the road lies while off it. Latched -- see
        # recovery_signal() for why it must not be recomputed each frame.
        self._recovery_direction = None

        # Last signal with a usable reading, so a blind frame keeps steering
        # the way it was rather than snapping straight.
        self._last_signal = 0.0

    def _row_offset(self, reading, half_width, target_clearance=None):
        """Where the car sits across the road at one row, in [-1, 1].

        Positive means steer right, negative means steer left.

        With both walls visible the car is placed between them, normalised
        by the road width at that row -- so +-1 means hard against a wall.
        (Normalising by the frame instead caps this at +-0.5 on a road that
        fills the screen, which puts the upper thresholds out of reach.)

        With only one wall visible the road runs off-screen that way, and
        there is no centre to aim at. Substituting half_width for the open
        side invents a large offset and makes the car flap between full-lock
        left and right, so instead it only reacts when the wall it CAN see
        is closer than target_clearance, and then steers away from it.
        """
        if target_clearance is None:
            target_clearance = self.target_clearance

        if not reading.get("on_road", True):
            # Off the track entirely. Both edges measure zero here, which
            # otherwise reads as a perfectly centred car -- and the
            # controller then drove straight into a wall at full throttle
            # indefinitely. Two runs lost 59% of their frames to exactly
            # this. Steer hard toward wherever the road actually is.
            nearest = reading.get("nearest_road_x")
            if nearest is None:
                return 0.0, False

            centre = reading.get("center_x", half_width)
            if nearest == centre:
                return 0.0, False

            direction = 1.0 if nearest > centre else -1.0
            return direction * OFF_ROAD_DEMAND, False

        left = reading["left_px"]
        right = reading["right_px"]

        if left is None and right is None:
            # Road fills the view. No correction to make, but this row also
            # says nothing about where the road goes, so it must not be used
            # as a curvature reference -- see the second return value.
            return 0.0, False

        if left is not None and right is not None:
            road_width = left + right
            if road_width <= 0:
                return 0.0, True
            return (right - left) / road_width, True

        if left is None:
            # Wall on the right only.
            if right >= target_clearance:
                return 0.0, True
            return -(target_clearance - right) / target_clearance, True

        # Wall on the left only.
        if left >= target_clearance:
            return 0.0, True
        return (target_clearance - left) / target_clearance, True

    def recovery_signal(self, readings):
        """Latched steering demand back onto the road, or None if on it.

        The direction is decided ONCE and held until the machine is back.
        Recomputing it every frame does not work: measured off the road, the
        reported side of the road flips on 18% of frames, so the smoothing
        averages the contradictory answers to nearly nothing and the machine
        coasts off-track until it stops. A stationary machine cannot steer,
        so it never gets back.
        """
        if not readings:
            return None

        near = readings[0]
        if near.get("on_road", True):
            self._recovery_direction = None
            return None

        if self._recovery_direction is None:
            # Decide once, pooling every row that can see the road so a
            # single noisy row cannot pick the direction on its own.
            vote = 0
            for reading in readings:
                nearest = reading.get("nearest_road_x")
                if nearest is None:
                    continue
                centre = reading.get("center_x", 128)
                if nearest > centre:
                    vote += 1
                elif nearest < centre:
                    vote -= 1

            if vote == 0:
                return None

            self._recovery_direction = 1.0 if vote > 0 else -1.0

        return self._recovery_direction * OFF_ROAD_DEMAND

    def signal(self, readings, half_width):
        """Weighted steering demand, and how sharp the corner ahead looks.

        Severity is how far the road shifts between the near row and the
        rows ahead -- i.e. curvature -- not how off-centre the far rows are.
        Using the raw far offset would read a car hugging a wall down a
        straight as a corner and brake for nothing.
        """
        total = 0.0
        weight_used = 0.0
        offsets = []

        for i, reading in enumerate(readings):
            offset, informative = self._row_offset(reading, half_width)
            offsets.append((offset, informative))

            weight = (
                self.row_weights[i]
                if i < len(self.row_weights)
                else self.row_weights[-1]
            )
            total += offset * weight
            weight_used += weight

        if weight_used == 0.0:
            # No readings at all, e.g. an undecodable frame. Hold rather
            # than snapping the wheel straight.
            return self._last_signal, 0.0

        combined = total / weight_used
        self._last_signal = combined

        # Curvature only means something between two rows that can both see
        # a wall. Comparing against a row where the road simply runs off
        # screen reads as maximum curvature and brakes on every straight.
        severity = 0.0
        if offsets and offsets[0][1]:
            near = offsets[0][0]
            ahead = [abs(o - near) for o, informative in offsets[1:] if informative]
            severity = max(ahead) if ahead else 0.0

        return combined, severity

    def update(self, readings, half_width, speed):
        """Returns ((steer, throttle, shoulder), reason)."""
        recovery = self.recovery_signal(readings)

        if recovery is not None:
            # Off the road: getting back is the only thing that matters, and
            # it needs speed to steer at all, so stay on the throttle.
            self._smoothed = recovery
            self._last_signal = recovery
            self.last_severity = 0.0
            self.last_target = TOP_SPEED_RAW

            direction = STEER_RIGHT if recovery > 0 else STEER_LEFT
            shoulder = self._shoulder(recovery)
            side = "right" if recovery > 0 else "left"
            return (direction, THROTTLE_ACCEL, shoulder), f"recovering {side}"

        signal, severity = self.signal(readings, half_width)
        self.last_severity = severity

        steer, note = self._steer(signal)

        # Lean off the smoothed demand as well, so the shoulder button does
        # not chatter on noise the steering has already filtered out.
        shoulder = self._shoulder(self._smoothed)
        throttle, throttle_note = self._throttle(severity, speed)

        return (steer, throttle, shoulder), f"{note}, {throttle_note}"

    def _steer(self, raw_signal):
        # Two things stop the wheel flapping. First the demand is smoothed,
        # so a single noisy frame cannot swing it to full lock. Second a
        # direction, once taken, is held briefly -- measured 2.1 reversals a
        # second before this, which scrubs speed and unsettles the car.
        self._smoothed += self.smoothing * (raw_signal - self._smoothed)
        signal = self._smoothed

        magnitude = abs(signal)

        if magnitude < self.deadzone:
            # Not drifting enough to correct; let the duty accumulator decay
            # so a stale fraction doesn't fire a stray input later.
            self._duty = 0.0
            self._hold = 0
            return STEER_STRAIGHT, "straight"

        wanted = STEER_RIGHT if signal > 0 else STEER_LEFT

        if (
            self._last_direction not in (None, STEER_STRAIGHT)
            and wanted != self._last_direction
            and self._hold < self.reversal_hold
            and magnitude < self.full_scale
        ):
            # Wants to reverse, but not urgently and not long enough since
            # the last change. Run straight instead of snapping back.
            self._hold += 1
            self._duty = 0.0
            return STEER_STRAIGHT, f"hold {self._hold}"

        if wanted != self._last_direction:
            self._hold = 0
        else:
            self._hold += 1

        demand = min(magnitude / self.full_scale, 1.0)
        self._duty += demand

        if self._duty < 1.0:
            return STEER_STRAIGHT, f"ease {demand:.2f}"

        self._duty -= 1.0
        self._last_direction = wanted
        return wanted, f"{'right' if signal > 0 else 'left'} {demand:.2f}"

    def _shoulder(self, signal):
        if not self.allow_lean or abs(signal) < self.lean_threshold:
            return SHOULDER_NONE
        return SHOULDER_R if signal > 0 else SHOULDER_L

    def target_speed(self, severity):
        """Speed the upcoming curvature can be carried through."""
        if severity <= STRAIGHT_SEVERITY:
            return TOP_SPEED_RAW

        excess = severity - STRAIGHT_SEVERITY
        target = TOP_SPEED_RAW * (1.0 - self.curve_slowdown * excess)
        return max(target, self.min_corner_speed)

    def _throttle(self, severity, speed):
        target = self.target_speed(severity)
        self.last_target = target

        if speed is None:
            return THROTTLE_ACCEL, "accel"

        if speed <= target:
            # Under target: back on the throttle immediately. Coasting on
            # past this point is what loses the time.
            return THROTTLE_ACCEL, f"accel (target {target:.0f})"

        # One coast frame sheds COAST_DECEL_PER_FRAME, so if the excess is
        # smaller than that a single frame would overshoot below target.
        # Stay on the throttle and let the corner scrub it instead.
        if speed - target < COAST_DECEL_PER_FRAME:
            return THROTTLE_ACCEL, f"accel (near target {target:.0f})"

        return THROTTLE_COAST, f"coast {speed:.0f}>{target:.0f}"
