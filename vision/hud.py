"""Read the F-Zero HUD from the frame.

Power is the quantity that actually ends a run: it drains on every wall
scrape and the machine is destroyed when it reaches zero. Wall contact was
being used as a proxy for it, but the bar is right there on screen, so it
can simply be measured.

Confirmed against captured frames: power reads ~99% over the first ten
checkpoints, 68% by checkpoint 20-30, and 15% by 30-45 -- and the frames
reading zero are exactly the checkpoints where runs were ending.
"""

import numpy as np

# The bar sits in the top-right of the HUD. Found by scanning captured
# frames for its fill colour; these are native 256x224 coordinates.
POWER_ROW = 22
POWER_X0 = 176
POWER_X1 = 239
POWER_WIDTH = POWER_X1 - POWER_X0 + 1

# The fill is dithered, so counting coloured pixels tops out around half
# even on a full bar. The bar empties from the right, so the position of
# the rightmost fill pixel is the level -- and that is immune to the
# dithering.
FILL_MIN_RED = 150
FILL_RED_OVER_GREEN = 60


def _fill_mask(frame_bgr, row):
    strip = frame_bgr[row, POWER_X0 : POWER_X1 + 1]
    blue = strip[:, 0].astype(np.int16)
    green = strip[:, 1].astype(np.int16)
    red = strip[:, 2].astype(np.int16)
    return (red > FILL_MIN_RED) & (red - green > FILL_RED_OVER_GREEN)


def read_power(frame_bgr):
    """Power remaining, 0.0 to 1.0, or None if the bar is not on screen.

    None means this is not a race view -- a results screen, for instance --
    rather than an empty bar, which reads 0.0.
    """
    if frame_bgr is None or frame_bgr.shape[0] <= POWER_ROW:
        return None

    mask = _fill_mask(frame_bgr, POWER_ROW)
    filled = np.flatnonzero(mask)

    if filled.size == 0:
        # Could be an empty bar or no bar at all. The bar's white surround
        # is still drawn when empty, so look for that to tell them apart.
        surround = frame_bgr[POWER_ROW - 3, POWER_X0 : POWER_X1 + 1]
        if surround.min(axis=1).mean() > 120:
            return 0.0
        return None

    return float(filled.max() + 1) / POWER_WIDTH


# The game FLASHES the power bar when the machine takes damage, so the raw
# reading dips to zero and back many times a second. Counting every dip as
# damage reported 29.6 bars of drain over a run that really lost about two.
#
# A flash only ever makes the bar look emptier than it is, so the maximum
# over a short window is the true level.
POWER_FLASH_WINDOW = 12


class PowerTracker:
    """Smooths the flashing power bar into a usable level and drain total."""

    def __init__(self, window=POWER_FLASH_WINDOW):
        self.window = window
        self.level = None
        self.drained = 0.0
        self._recent = []

    def update(self, frame_bgr):
        """Feed a frame; returns the smoothed power level, or None."""
        raw = read_power(frame_bgr)

        if raw is None:
            # Not a race view. Drop the history so the next life starts
            # clean rather than comparing against the previous one.
            self._recent.clear()
            self.level = None
            return None

        self._recent.append(raw)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        smoothed = max(self._recent)

        if self.level is not None and smoothed < self.level:
            self.drained += self.level - smoothed

        self.level = smoothed
        return smoothed

    def reset(self):
        """Forget history, e.g. after loading a savestate."""
        self._recent.clear()
        self.level = None
