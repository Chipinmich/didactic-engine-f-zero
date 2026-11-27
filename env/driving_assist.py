"""Geometry-based guard rails shared by training and evaluation.

The genetic policy still chooses steering and lean.  This module only steps
in when the vision readings predict a wall strike, when the policy asks to
run straight while clearly off-centre, or when it starts flapping left/right
faster than the machine can settle.  Keeping the guard in the environment
means genomes are evolved around the same safety system used at evaluation.
"""

from bridge_client import (
    SHOULDER_L,
    SHOULDER_NONE,
    SHOULDER_R,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
)

WARNING_CLEARANCE = 52.0

# Wall clearance (pixels) below which "emergency" mode steps in at all. This
# is the ONLY intervention left once mode="emergency" strips out the
# predicted-wall, lookahead-turn and centring triggers -- so it is also the
# only knob left to tune the safety-intervention rate under that mode.
#
# Was 10.0, lowered to reduce how often mode="emergency" steps in at all --
# it is the only intervention left once that mode strips out the
# predicted-wall, lookahead-turn and centring triggers, so it is also the
# only knob available to tune the safety-intervention rate under it.
#
# BE WARNED before trusting any specific target number here: measuring this
# against a DQN checkpoint (12.5k training steps) and a genetic policy
# (3 generations, population 6) -- both quick smoke-test checkpoints, not
# properly trained models -- gave wildly different readings run to run. A
# sweep at clearance 4.0 measured DQN override 9.2% and genetic 3.2%; two
# immediate repeats of the exact same configuration then measured 27.7%/
# 15.3% and 29.7%/18.7%. Same code path, same constant, same models -- nothing
# else changed. That is not noise on the scale this project's own
# compare_configs.py warns about (81 vs 66); it means an undertrained,
# fragile model's behaviour swings hard across separate emulator sessions,
# and neither number above is a number to design around.
#
# What IS real: lower clearance monotonically reduced intervention in the
# sweep (14.4/11.8/9.2 for DQN at 10/6/4), and mode="emergency" itself cuts
# intervention drastically versus mode="full" (roughly 87%->30% for DQN,
# 43%->20% for the genetic policy, across several measurements). 4.0 is kept
# as a reasonable middle value on that trend, not as a calibrated target.
# Re-measure this against models that were actually trained to convergence,
# with several repeats each, before treating any percentage from this
# constant as settled.
EMERGENCY_CLEARANCE = 4.0
TARGET_CLEARANCE = 55.0
PREDICTION_FRAMES = 8
REVERSAL_HOLD_FRAMES = 6
CENTRE_DEADZONE = 0.16
TURN_STEER_THRESHOLD = 0.12
TURN_LEAN_THRESHOLD = 0.22

# When to hand control back to the policy early, i.e. hold the action for
# fewer than frame_skip frames. Separate from the two thresholds above on
# purpose -- see action_hold() for what happened when they were shared.
#
# Set against the measured distributions over 47180 frames of a live run:
# |curve_signal| runs median 0.500 / 90th percentile 1.000, and wall risk is
# bimodal at median 0.000 / mean 0.393. So these sit high enough to mean
# "unusually urgent" rather than "normal driving".
URGENT_HOLD_RISK = 0.75
URGENT_HOLD_CURVE = 0.90
BUSY_HOLD_RISK = 0.45
BUSY_HOLD_CURVE = 0.55

# vision.SCAN_ROWS is far-to-near.  The near row protects the machine's
# current position; the other rows add enough look-ahead to enter corners
# before the near edge becomes dangerous.
ROW_WEIGHTS = (0.45, 0.60, 0.80, 1.00)


def _row_signal(reading, target_clearance=TARGET_CLEARANCE):
    """Steering demand in [-1, 1]; positive means move right."""
    if not reading.get("on_road", True):
        nearest = reading.get("nearest_road_x")
        centre = reading.get("center_x", 128)
        if nearest is None or nearest == centre:
            return 0.0
        return 1.0 if nearest > centre else -1.0

    left = reading.get("left_px")
    right = reading.get("right_px")
    if left is None and right is None:
        return 0.0
    if left is not None and right is not None:
        width = left + right
        return 0.0 if width <= 0 else max(min((right - left) / width, 1.0), -1.0)
    if left is None:
        return -max((target_clearance - right) / target_clearance, 0.0)
    return max((target_clearance - left) / target_clearance, 0.0)


def centre_signal(readings):
    """Pool current position and upcoming road shape into one demand."""
    if not readings:
        return 0.0
    total = 0.0
    used = 0.0
    for index, reading in enumerate(readings):
        weight = ROW_WEIGHTS[min(index, len(ROW_WEIGHTS) - 1)]
        total += _row_signal(reading) * weight
        used += weight
    return max(min(total / used, 1.0), -1.0) if used else 0.0


def curve_ahead_signal(readings):
    """Upcoming road shift relative to the car's current track position.

    A raw far-row offset is not enough: a car hugging a wall on a straight
    produces the same offset at every row.  Subtracting the two near rows
    leaves curvature, and on the recorded first turn this becomes visible
    around checkpoints 17-25 while near-wall risk does not spike until 30.
    """
    if len(readings or []) < 2:
        return 0.0
    offsets = [_row_signal(reading) for reading in readings]
    split = max(len(offsets) // 2, 1)
    ahead = sum(offsets[:split]) / split
    near = sum(offsets[split:]) / max(len(offsets) - split, 1)
    return max(min(ahead - near, 1.0), -1.0)


class DrivingAssist:
    """Predict wall approach, centre gently, and debounce reversals."""

    def __init__(
        self,
        enabled=True,
        warning_clearance=WARNING_CLEARANCE,
        emergency_clearance=EMERGENCY_CLEARANCE,
        prediction_frames=PREDICTION_FRAMES,
        reversal_hold=REVERSAL_HOLD_FRAMES,
        centre_deadzone=CENTRE_DEADZONE,
        turn_steer_threshold=TURN_STEER_THRESHOLD,
        turn_lean_threshold=TURN_LEAN_THRESHOLD,
        mode="full",
    ):
        self.enabled = bool(enabled)
        if mode not in ("full", "emergency"):
            raise ValueError("assist mode must be 'full' or 'emergency'")
        self.mode = mode
        self.warning_clearance = float(warning_clearance)
        self.emergency_clearance = float(emergency_clearance)
        self.prediction_frames = max(int(prediction_frames), 0)
        self.reversal_hold = max(int(reversal_hold), 0)
        self.centre_deadzone = float(centre_deadzone)
        self.turn_steer_threshold = float(turn_steer_threshold)
        self.turn_lean_threshold = float(turn_lean_threshold)
        self.reset()

    def reset(self):
        self._previous = {"left": None, "right": None}
        self._closing = {"left": 0.0, "right": 0.0}
        self._last_direction = STEER_STRAIGHT
        self._direction_frames = self.reversal_hold
        self.state = {
            "risk": 0.0,
            "signal": 0.0,
            "curve_signal": 0.0,
            "direction": STEER_STRAIGHT,
            "emergency": False,
            "predicted_left": None,
            "predicted_right": None,
            "off_road": False,
        }

    def _predict(self, side, clearance):
        if clearance is None:
            self._previous[side] = None
            self._closing[side] = 0.0
            return None
        clearance = float(clearance)
        previous = self._previous[side]
        raw_closing = 0.0 if previous is None else max(previous - clearance, 0.0)
        # Smooth pixel-mask jitter, and cap a single bad reading so it cannot
        # claim a wall is hundreds of pixels closer on the next frame.
        self._closing[side] = min(
            12.0, 0.5 * self._closing[side] + 0.5 * raw_closing
        )
        self._previous[side] = clearance
        return clearance - self._closing[side] * self.prediction_frames

    def observe(self, readings):
        signal = centre_signal(readings)
        curve_signal = curve_ahead_signal(readings)
        near = readings[-1] if readings else {}
        off_road = bool(readings) and not near.get("on_road", True)
        predicted_left = self._predict("left", near.get("left_px"))
        predicted_right = self._predict("right", near.get("right_px"))

        visible = [
            (value, direction)
            for value, direction in (
                (predicted_left, STEER_RIGHT),
                (predicted_right, STEER_LEFT),
            )
            if value is not None
        ]
        if visible:
            closest, direction = min(visible, key=lambda pair: pair[0])
            span = max(self.warning_clearance - self.emergency_clearance, 1.0)
            risk = min(max((self.warning_clearance - closest) / span, 0.0), 1.0)
        else:
            closest, direction, risk = None, STEER_STRAIGHT, 0.0

        if off_road:
            risk = 1.0
            if signal > 0:
                direction = STEER_RIGHT
            elif signal < 0:
                direction = STEER_LEFT

        self.state = {
            "risk": risk,
            "signal": signal,
            "curve_signal": curve_signal,
            "direction": direction,
            "emergency": off_road or (closest is not None and closest <= self.emergency_clearance),
            "predicted_left": predicted_left,
            "predicted_right": predicted_right,
            "off_road": off_road,
        }
        return dict(self.state)

    def action_hold(self, normal_frames):
        """How many frames the next policy decision may safely be held.

        These thresholds are deliberately NOT turn_steer_threshold and
        turn_lean_threshold. Those two decide whether to steer or to lean, and
        are correctly tuned for that -- but they were also used here, and as a
        trigger for "cut the hold to a single frame" they mean "always":
        measured over 47180 frames of a live run, |curve_signal| has a MEDIAN
        of 0.500 against a lean threshold of 0.22, so that branch fired on 79%
        of frames. Together with risk >= 0.20 it collapsed **91.9% of policy
        decisions to exactly one frame**, against a nominal frame skip of 4.

        That is the mechanism behind steering that visibly flickers instead of
        being held. The policy re-decided at 60Hz and changed its mind on 45%
        of frames, so the pad saw a different direction almost every frame.
        Steering authority itself is fine -- a held input pulls the machine 35
        world units off line within 5 frames and 412 within 59, while it
        travels 567 forward -- but only if the input is actually held.

        One frame is now reserved for a genuine emergency: off the road, or a
        wall inside emergency_clearance. Everything else gets at least two, and
        an untroubled frame gets the full skip. On the same recorded
        distributions this yields a mean hold near 2.7 rather than 1.12.
        """
        normal_frames = max(int(normal_frames), 1)
        if self.state["emergency"]:
            return 1
        if (
            self.state["risk"] >= URGENT_HOLD_RISK
            or abs(self.state["curve_signal"]) >= URGENT_HOLD_CURVE
        ):
            return min(normal_frames, 2)
        if (
            self.state["risk"] >= BUSY_HOLD_RISK
            or abs(self.state["curve_signal"]) >= BUSY_HOLD_CURVE
        ):
            return min(normal_frames, 3)
        return normal_frames

    @staticmethod
    def _matching_shoulder(steer):
        if steer == STEER_LEFT:
            return SHOULDER_L
        if steer == STEER_RIGHT:
            return SHOULDER_R
        return SHOULDER_NONE

    def apply(self, steer, shoulder):
        """Return ((steer, shoulder), diagnostics) for one emulated frame."""
        requested = (int(steer), int(shoulder))
        if not self.enabled:
            return requested, {
                "override": False,
                "reason": "disabled",
                "reversal_blocked": False,
                **self.state,
            }

        steer, shoulder = requested
        risk = self.state["risk"]
        signal = self.state["signal"]
        curve_signal = self.state["curve_signal"]
        direction = self.state["direction"]
        reason = "policy"

        # Genetic v4 uses the assist only as a last-resort crash guard.  The
        # warning, look-ahead and centring signals remain in the observation
        # and diagnostics, but the genome must earn credit for acting on
        # them itself.  Full mode is retained for legacy PPO/DQN/controller
        # runs so this change cannot silently alter a separate experiment.
        if self.mode == "emergency":
            if self.state["emergency"] and direction != STEER_STRAIGHT:
                applied = (direction, self._matching_shoulder(direction))
                reason = "emergency"
            else:
                applied = requested
            if applied[0] in (STEER_LEFT, STEER_RIGHT):
                self._last_direction = applied[0]
                self._direction_frames += 1
            else:
                self._direction_frames += 1
            return applied, {
                "override": applied != requested,
                "reason": reason,
                "reversal_blocked": False,
                **self.state,
            }

        if self.state["emergency"] and direction != STEER_STRAIGHT:
            steer = direction
            shoulder = self._matching_shoulder(direction)
            reason = "emergency"
        elif risk >= 0.35 and direction != STEER_STRAIGHT:
            steer = direction
            # An opposite lean fights the escape direction.  Matching lean is
            # retained; neutral is used otherwise to avoid needless speed loss.
            shoulder = (
                shoulder
                if shoulder == self._matching_shoulder(direction)
                else SHOULDER_NONE
            )
            reason = "predicted_wall"
        elif abs(curve_signal) >= self.turn_lean_threshold:
            steer = STEER_RIGHT if curve_signal > 0 else STEER_LEFT
            shoulder = self._matching_shoulder(steer)
            reason = "lookahead_turn"
        elif abs(curve_signal) >= self.turn_steer_threshold:
            wanted = STEER_RIGHT if curve_signal > 0 else STEER_LEFT
            if steer == STEER_STRAIGHT or steer != wanted:
                steer = wanted
                if shoulder not in (SHOULDER_NONE, self._matching_shoulder(wanted)):
                    shoulder = SHOULDER_NONE
                reason = "lookahead_turn"
        elif steer == STEER_STRAIGHT and abs(signal) >= self.centre_deadzone:
            steer = STEER_RIGHT if signal > 0 else STEER_LEFT
            reason = "centre"

        reversal_blocked = False
        if steer in (STEER_LEFT, STEER_RIGHT):
            urgent = risk >= 0.75 or abs(signal) >= 0.55
            if (
                self._last_direction in (STEER_LEFT, STEER_RIGHT)
                and steer != self._last_direction
                and self._direction_frames < self.reversal_hold
                and not urgent
            ):
                steer = STEER_STRAIGHT
                shoulder = SHOULDER_NONE
                reversal_blocked = True
                reason = "reversal_hold"

        if steer in (STEER_LEFT, STEER_RIGHT):
            if steer == self._last_direction:
                self._direction_frames += 1
            else:
                self._last_direction = steer
                self._direction_frames = 1
        else:
            self._direction_frames += 1

        applied = (steer, shoulder)
        return applied, {
            "override": applied != requested,
            "reason": reason,
            "reversal_blocked": reversal_blocked,
            **self.state,
        }
