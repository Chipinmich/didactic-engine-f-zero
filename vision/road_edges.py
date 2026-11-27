"""Vision-only road edge detection for F-Zero (SNES).

No RAM/memory-address reading is involved: everything here works purely off
the frame BizHawk's Lua bridge hands over. The road surface is a
low-saturation grey, which is enough to tell it apart from the car sprite,
track barriers, crowd, and sky via an HSV threshold.

Two consumers share this module:
  - autopilot.py       measure_edges()     pixel distance to each wall
  - env/fzero_env.py   extract_features()  normalised RL observation

Both go through the same scanning code so their view of the road cannot
drift apart. The scan is vectorised and, for the RL path, only the scan rows
are converted to HSV rather than the whole frame -- this runs once per
emulated frame, so it is on the hot path.
"""

import cv2
import numpy as np
import math

# Fixed-position HUD elements (mini-map, rank/lap counter) that sit low in
# the frame near where the "near" scan row lands. They aren't road-colored,
# so without this they get mistaken for a track wall. Boxes are (x1, y1, x2,
# y2) in native 256x224 screenshot coordinates; treated as "road" (ignored)
# during edge scanning. Recalibrate if the HUD layout ever changes.
HUD_IGNORE_BOXES = [
    (0, 135, 46, 224),  # mini-map, bottom-left
    (200, 200, 256, 224),  # rank/lap counter, bottom-right
]

# The road is dithered from two alternating colours, and on a real Mute City
# frame those measure (S=26, V=165) for ~93% of road pixels and (S=85, V=99)
# for the rest. Thresholds must cover BOTH: sat_max=50/v_min=110 catches only
# the first, leaving the second scattered through the road as false walls,
# which shreds the mask and reports a wall right beside the car.
#
# These values cover 100% of a measured road patch. Off-track sits at V~63
# (excluded by v_min) and the city skyline reaches S~165 (excluded by
# sat_max).
#
# NOTE: still track-specific. Big Blue / Sand Ocean / Death Wind use
# different palettes and will need re-tuning. Use `autopilot.py --calibrate`
# to find values for a new track.
DEFAULT_SAT_MAX = 90
DEFAULT_V_MIN = 95
DEFAULT_V_MAX = 205

# Rows sampled to build the RL observation, far to near. The last one is the
# row autopilot.py drives on.
SCAN_ROWS = (75, 105, 140, 183)

# Seven sparse perspective rays originate just above the car sprite and fan
# forward through the road.  They add early curve shape without converting
# the full 256x224 frame to HSV: only roughly a thousand sampled pixels are
# classified per frame, keeping this suitable for eight live emulators.
RAY_ANGLES = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0)
RAY_ORIGIN_Y = 150
NUM_RAY_FEATURES = len(RAY_ANGLES)

# Half-width of the car sprite, skipped when scanning outward so the car
# itself is never mistaken for a wall. Measured on a real frame: the sprite
# spans y=155..190 and reaches 23px from centre at its widest, so 20 starts
# the scan just inside the car and reports a wall 0px away. 28 leaves margin.
DEFAULT_CAR_HALF_WIDTH = 28

# Consecutive non-road pixels required to call something a wall. The road is
# dithered, so a single stray pixel is not an edge.
DEFAULT_MIN_RUN = 2

# Per row: gap to the left wall, gap to the right wall, and whether there is
# road under the machine at all. Then two values saying which way the road
# lies when it has left the track, speed, and seven forward ray distances.
#
# The on-road flags and the direction are not optional detail. Without them
# a machine wedged between two close walls, one off-track with the road far
# to its left, and one off-track with the road far to its RIGHT all produce
# the identical observation of zeros -- and two of those need opposite
# actions. The agent could not learn to get back on the track because it
# could not tell it had left it.
FEATURES_PER_ROW = 3
ROW_FEATURES = len(SCAN_ROWS) * FEATURES_PER_ROW
ROAD_LEFT_INDEX = ROW_FEATURES
ROAD_RIGHT_INDEX = ROW_FEATURES + 1
SPEED_INDEX = ROW_FEATURES + 2
RAY_START_INDEX = ROW_FEATURES + 3
LEGACY_NUM_FEATURES = RAY_START_INDEX
NUM_FEATURES = LEGACY_NUM_FEATURES + NUM_RAY_FEATURES


def road_mask(
    frame_bgr,
    sat_max=DEFAULT_SAT_MAX,
    v_min=DEFAULT_V_MIN,
    v_max=DEFAULT_V_MAX,
    ignore_boxes=HUD_IGNORE_BOXES,
    rows=None,
):
    """Boolean mask that is True where a pixel looks like road.

    Returns (H, W) for the whole frame, or (len(rows), W) when `rows` is
    given -- the fast path, since converting three or four rows to HSV is far
    cheaper than the entire frame.

    `ignore_boxes` are treated as road (i.e. scanned straight through) so
    fixed HUD overlays don't get mistaken for track edges.
    """
    if rows is None:
        source = frame_bgr
        row_indices = None
    else:
        row_indices = list(rows)
        source = frame_bgr[row_indices, :, :]

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s <= sat_max) & (v >= v_min) & (v <= v_max)

    for x1, y1, x2, y2 in ignore_boxes or []:
        if row_indices is None:
            mask[y1:y2, x1:x2] = True
        else:
            for i, row in enumerate(row_indices):
                if y1 <= row < y2:
                    mask[i, x1:x2] = True

    return mask


def _first_run_start(flags, min_run):
    """Index where the first run of `min_run` consecutive True values begins."""
    if flags.size < min_run:
        return None

    if min_run <= 1:
        hits = np.flatnonzero(flags)
        return int(hits[0]) if hits.size else None

    windows = np.convolve(flags.astype(np.int32), np.ones(min_run, np.int32), "valid")
    hits = np.flatnonzero(windows == min_run)
    return int(hits[0]) if hits.size else None


def _scan_row(row_is_road, center_x, car_half_width, min_run):
    """Scan outward from the car for the first wall on each side.

    Returns (left_edge_x, right_edge_x, left_start, right_start, on_road,
    nearest_road_x). An edge is None if the scan reached the frame boundary
    without finding one (i.e. the road is clear that way).

    on_road is False when there is no road under the car at all -- it has
    left the track. That has to be distinguishable from being centred:
    both edges then measure zero, which reads as a perfectly centred car and
    had the controller driving straight into a wall indefinitely.
    nearest_road_x then says which way the road is.
    """
    width = row_is_road.size
    left_start = max(center_x - car_half_width, 0)
    right_start = min(center_x + car_half_width, width - 1)

    not_road = ~row_is_road

    # Walking left, so reverse the slice; index r maps back to left_start - r.
    leftward = not_road[left_start::-1]
    r = _first_run_start(leftward, min_run)
    left_edge_x = left_start - r if r is not None else None

    rightward = not_road[right_start:]
    r = _first_run_start(rightward, min_run)
    right_edge_x = right_start + r if r is not None else None

    on_road = bool(row_is_road[left_start : right_start + 1].any())

    nearest_road_x = None
    if not on_road:
        road_columns = np.flatnonzero(row_is_road)
        if road_columns.size:
            nearest_road_x = int(
                road_columns[np.argmin(np.abs(road_columns - center_x))]
            )

    return left_edge_x, right_edge_x, left_start, right_start, on_road, nearest_road_x


def measure_edges(
    frame_bgr,
    scan_y,
    center_x=None,
    car_half_width=DEFAULT_CAR_HALF_WIDTH,
    sat_max=DEFAULT_SAT_MAX,
    v_min=DEFAULT_V_MIN,
    v_max=DEFAULT_V_MAX,
    min_run=DEFAULT_MIN_RUN,
    ignore_boxes=HUD_IGNORE_BOXES,
):
    """Measure the car's distance (in pixels) to the nearest track edge to
    its left and right, at a given scan row.

    Returns a dict with left_px/right_px (None if no edge found before the
    frame boundary, i.e. the road is clear that direction) plus the raw mask
    and edge x-coordinates, useful for drawing a calibration overlay.
    """
    h, w = frame_bgr.shape[:2]
    if center_x is None:
        center_x = w // 2

    mask = road_mask(frame_bgr, sat_max, v_min, v_max, ignore_boxes)
    (
        left_edge_x,
        right_edge_x,
        left_start,
        right_start,
        on_road,
        nearest_road_x,
    ) = _scan_row(mask[scan_y], center_x, car_half_width, min_run)

    return {
        "on_road": on_road,
        "nearest_road_x": nearest_road_x,
        "left_px": (left_start - left_edge_x) if left_edge_x is not None else None,
        "right_px": (right_edge_x - right_start) if right_edge_x is not None else None,
        "left_edge_x": left_edge_x,
        "right_edge_x": right_edge_x,
        "left_start": left_start,
        "right_start": right_start,
        "scan_y": scan_y,
        "center_x": center_x,
        "mask": mask,
    }


def measure_rows(
    frame_bgr,
    scan_ys,
    center_x=None,
    car_half_width=DEFAULT_CAR_HALF_WIDTH,
    sat_max=DEFAULT_SAT_MAX,
    v_min=DEFAULT_V_MIN,
    v_max=DEFAULT_V_MAX,
    min_run=DEFAULT_MIN_RUN,
    ignore_boxes=HUD_IGNORE_BOXES,
):
    """measure_edges() for several rows at once, sharing one mask.

    The driving loop needs a near row plus lookahead rows every frame, and
    calling measure_edges() per row would rebuild a full-frame mask each
    time. This converts only the requested rows to HSV.
    """
    h, w = frame_bgr.shape[:2]
    if center_x is None:
        center_x = w // 2

    rows = list(scan_ys)
    mask = road_mask(frame_bgr, sat_max, v_min, v_max, ignore_boxes, rows=rows)

    readings = []
    for i, scan_y in enumerate(rows):
        (
            left_x,
            right_x,
            left_start,
            right_start,
            on_road,
            nearest_road_x,
        ) = _scan_row(mask[i], center_x, car_half_width, min_run)
        readings.append(
            {
                "on_road": on_road,
                "nearest_road_x": nearest_road_x,
                "road_pixels": int(mask[i].sum()),
                "left_px": (left_start - left_x) if left_x is not None else None,
                "right_px": (right_x - right_start) if right_x is not None else None,
                "left_edge_x": left_x,
                "right_edge_x": right_x,
                "left_start": left_start,
                "right_start": right_start,
                "scan_y": scan_y,
                "center_x": center_x,
                "mask": None,
            }
        )

    return readings


def _ray_path(height, width, origin, angle):
    """Integer pixel coordinates from the car toward one image boundary."""
    radians = math.radians(float(angle))
    dx = math.sin(radians)
    dy = -math.cos(radians)
    x0, y0 = origin
    limit = int(math.ceil(math.hypot(width, height))) + 1
    points = []
    previous = None
    for distance in range(limit):
        x = int(round(x0 + dx * distance))
        y = int(round(y0 + dy * distance))
        if not (0 <= x < width and 0 <= y < height):
            break
        point = (x, y)
        if point != previous:
            points.append(point)
            previous = point
    return points


def measure_rays(
    frame_bgr,
    angles=RAY_ANGLES,
    origin=None,
    sat_max=DEFAULT_SAT_MAX,
    v_min=DEFAULT_V_MIN,
    v_max=DEFAULT_V_MAX,
    min_run=DEFAULT_MIN_RUN,
):
    """Measure normalized road distance along seven rays ahead of the car.

    Distance 1.0 means road continued to the image boundary; 0.0 means a
    wall began at the ray origin.  Each result includes its endpoint for a
    debug overlay.  Sampling only the ray pixels is much cheaper than making
    a full-frame road mask.
    """
    if frame_bgr is None:
        return []
    height, width = frame_bgr.shape[:2]
    if origin is None:
        origin = (width // 2, min(RAY_ORIGIN_Y, height - 1))

    results = []
    for angle in angles:
        path = _ray_path(height, width, origin, angle)
        if not path:
            continue
        xs = np.fromiter((point[0] for point in path), dtype=np.int32)
        ys = np.fromiter((point[1] for point in path), dtype=np.int32)
        pixels = frame_bgr[ys, xs].reshape(-1, 1, 3)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        is_road = (
            (hsv[:, 1] <= sat_max)
            & (hsv[:, 2] >= v_min)
            & (hsv[:, 2] <= v_max)
        )
        hit = _first_run_start(~is_road, min_run)
        last = len(path) - 1 if hit is None else max(hit - 1, 0)
        maximum = max(len(path) - 1, 1)
        results.append(
            {
                "angle": float(angle),
                "distance": float(np.clip(last / maximum, 0.0, 1.0)),
                "origin": tuple(origin),
                "end": tuple(path[last]),
                "hit": hit is not None,
            }
        )
    return results


def ray_distances(frame_bgr):
    """Fixed-width seven-ray feature vector in left-to-right order."""
    rays = measure_rays(frame_bgr)
    values = np.zeros(NUM_RAY_FEATURES, dtype=np.float32)
    for index, ray in enumerate(rays[:NUM_RAY_FEATURES]):
        values[index] = ray["distance"]
    return values


def features_from_readings(
    readings, speed_normalised, ray_values=None, half_width=128.0
):
    """Build the RL observation from readings already measured.

    Separate from extract_features so a caller that needs more than the
    observation -- the env wants to know whether the car is scraping a wall,
    to penalise it -- can measure once and use the result twice.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    road_left = 0.0
    road_right = 0.0

    for i, reading in enumerate(readings[: len(SCAN_ROWS)]):
        left = reading["left_px"]
        right = reading["right_px"]

        left_px = half_width if left is None else left
        right_px = half_width if right is None else right

        base = i * FEATURES_PER_ROW
        features[base] = min(left_px / half_width, 1.0)
        features[base + 1] = min(right_px / half_width, 1.0)

        on_road = bool(reading.get("on_road", True))
        features[base + 2] = 1.0 if on_road else 0.0

        # Which way the road lies, taken from the nearest row that can see
        # it. Split into two non-negative values so the observation stays
        # in [0, 1] and "no road that way" is plainly zero rather than a
        # midpoint the policy has to learn to read.
        if not on_road:
            nearest = reading.get("nearest_road_x")
            if nearest is not None:
                centre = reading.get("center_x", half_width)
                offset = (nearest - centre) / half_width
                if offset < 0:
                    road_left = max(road_left, min(-offset, 1.0))
                else:
                    road_right = max(road_right, min(offset, 1.0))

    features[ROAD_LEFT_INDEX] = road_left
    features[ROAD_RIGHT_INDEX] = road_right
    features[SPEED_INDEX] = speed_normalised
    if ray_values is not None:
        rays = np.asarray(ray_values, dtype=np.float32).ravel()
        used = min(len(rays), NUM_RAY_FEATURES)
        features[RAY_START_INDEX : RAY_START_INDEX + used] = np.clip(
            rays[:used], 0.0, 1.0
        )
    return features


def touching_wall(readings):
    """True if the car itself is scraping a wall.

    Only the nearest row counts -- SCAN_ROWS runs far to near, so that is
    the last one, at the car's own height. The lookahead rows are further
    down the track, where perspective squeezes the road into a narrow band,
    so a wall sits close to the centre in screen space even when the car is
    perfectly placed. Counting those made a random policy read as touching a
    wall on 75% of frames, which buried every other term in the reward.
    """
    if not readings:
        return False

    near = readings[-1]
    return near["left_px"] == 0 or near["right_px"] == 0


def extract_features(
    frame_bgr,
    speed_normalised,
    rows=SCAN_ROWS,
    car_half_width=DEFAULT_CAR_HALF_WIDTH,
    min_run=DEFAULT_MIN_RUN,
):
    """Returns row geometry, speed, and seven forward ray distances.

    All values are in [0, 1].

    A clear side -- no wall before the frame edge -- reads as 1.0, the
    maximum distance. A frame that cannot be decoded reads as all zeros,
    which the agent sees as walls pressing in on both sides.
    """
    if frame_bgr is None:
        return features_from_readings([], speed_normalised)
    readings = measure_rows(
        frame_bgr,
        rows,
        car_half_width=car_half_width,
        min_run=min_run,
    )
    return features_from_readings(
        readings, speed_normalised, ray_distances(frame_bgr)
    )


def draw_debug_overlay(frame_bgr, readings_list, rays=None):
    """Return a copy of frame_bgr with scan lines and detected edges drawn
    for one or more measure_edges() results (e.g. near + far rows)."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for readings in readings_list:
        y = readings["scan_y"]
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
        cv2.circle(out, (readings["center_x"], y), 3, (255, 255, 255), -1)

        if readings["left_edge_x"] is not None:
            cv2.circle(out, (readings["left_edge_x"], y), 4, (255, 0, 0), -1)
        if readings["right_edge_x"] is not None:
            cv2.circle(out, (readings["right_edge_x"], y), 4, (0, 0, 255), -1)

    for ray in rays if rays is not None else measure_rays(frame_bgr):
        distance = float(ray["distance"])
        colour = (0, int(255 * distance), int(255 * (1.0 - distance)))
        cv2.line(out, ray["origin"], ray["end"], colour, 1)
        cv2.circle(out, ray["end"], 2, (255, 255, 255), -1)

    return out
