"""Turn the machine's world position into a lap-progress checkpoint.

F-Zero keeps no lap-position counter in work RAM. The bridge used to read
0x000C1 as "progress"; that is the lap TIMER -- held still in mid-race it
counts on regardless, while the integral of speed does not move. See
tools/build_track_map.py for the measurements.

What the game does keep is the machine's world position, x at 0x00B70 and y at
0x00B90, both u16 little-endian. Progress is derived from those by finding the
nearest point on a centreline recorded from a real lap.

The search is FORWARD-BIASED rather than global. A global nearest-point lookup
breaks in two places: at the start/finish line, where checkpoint 89 and
checkpoint 0 are adjacent, so a car crossing the line would read as travelling
89 checkpoints backwards; and anywhere the track passes close to itself, where
a car on one side can be nearest to a checkpoint on the other. Searching a
window around the last known checkpoint removes both.
"""

import os

import numpy as np

DEFAULT_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "track_mute_city.npz")

# How far the search may look each way from the last checkpoint. Forward is
# generous because one policy step at racing speed can cover several
# checkpoints; backward is deliberately short, enough for a spin or a bounce
# off a wall but not enough to wrap round to the previous lap.
LOOK_AHEAD = 12
LOOK_BACK = 4


class TrackMap:
    def __init__(self, path=DEFAULT_MAP):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no track map at {path}. Record a lap with lua/ram_record.lua "
                "and run tools/build_track_map.py."
            )
        data = np.load(path)
        self.centreline = np.asarray(data["centreline"], dtype=np.float64)
        self.checkpoints = int(data["checkpoints"])

    def nearest(self, x, y):
        """Global nearest checkpoint. Only for the first reading of a lap."""
        offsets = self.centreline - np.array([float(x), float(y)])
        return int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))

    def advance(self, x, y, previous):
        """Nearest checkpoint within a window around `previous`.

        Returns (checkpoint, wrapped) where wrapped is True if the window ran
        past the end of the lap -- the caller uses it to notice a lap without
        having to trust the game's own counter timing.
        """
        if previous is None:
            return self.nearest(x, y), False

        count = self.checkpoints
        window = np.arange(previous - LOOK_BACK, previous + LOOK_AHEAD + 1)
        indices = window % count
        offsets = self.centreline[indices] - np.array([float(x), float(y)])
        best = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
        raw = int(window[best])
        return raw % count, raw >= count

    def distance_to_centre(self, x, y, checkpoint):
        offset = self.centreline[checkpoint % self.checkpoints] - np.array(
            [float(x), float(y)]
        )
        return float(np.hypot(*offset))
