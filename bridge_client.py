"""Python side of the EmuHawk bridge, shared by the RL env and the autopilot.

Protocol (see lua/bridge.lua):
    action.txt  "<action> <sequence>"        written here
    state.txt   "<speed> <resets> <frames> <lap> <lap_time>
                 <mmf_failures> <saves> <racing> <x> <y>
                 <speed_mode>"                              written by Lua
    frame       PNG bytes in a memory-mapped file

The bridge runs in lockstep: it advances one emulated frame per new sequence
number. So holding a button down means calling advance() repeatedly, not
sleeping -- a sleep would simply stall the emulator.
"""

import mmap
import os
import sys
import threading
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_FILE = os.path.join(BASE_DIR, "action.txt")
STATE_FILE = os.path.join(BASE_DIR, "state.txt")
FRAME_FILE = os.path.join(BASE_DIR, "frames", "frame.png")

MMF_NAME = "fzero_frame"


def instance_paths(instance=None):
    """Per-instance bridge endpoints.

    Training several emulators in parallel needs each one on its own files
    and its own shared-memory name; instance None is the single-emulator
    default that lua/bridge.lua uses out of the box.
    """
    if instance is None:
        return ACTION_FILE, STATE_FILE, FRAME_FILE, MMF_NAME

    return (
        os.path.join(BASE_DIR, f"action{instance}.txt"),
        os.path.join(BASE_DIR, f"state{instance}.txt"),
        os.path.join(BASE_DIR, "frames", f"frame{instance}.png"),
        f"{MMF_NAME}_{instance}",
    )

# BizHawk writes the raw PNG bytes into the mapping starting at offset 0 --
# there is no length prefix. (An earlier guess of a 4-byte little-endian
# length was wrong: those bytes read back as 0x474E5089, which is the PNG
# signature itself.) The decoder stops at IEND, so trailing bytes are
# harmless and the whole mapping can simply be handed to imdecode.
PNG_SIGNATURE = b"\x89PNG"

# The mapping is sized to the PNG, roughly 8-16KB for a 256x224 frame, and
# Windows fails MapViewOfFile when the requested size exceeds the mapping.
# The exact size is unknown and varies per frame, so it gets probed.
MMF_PAGE_SIZE = 4096
MMF_MAX_SIZE = 1 << 20

# Action space: three independent dimensions, because they genuinely are
# independent on the controller -- you can steer, throttle and lean at the
# same time. Exposed to PPO as MultiDiscrete([3, 3, 3]) and packed into a
# single integer for the file protocol, which bridge.lua unpacks.
#
# The shoulder buttons make the machine lean into a corner; that is the main
# technique for taking a tight turn at speed, and nothing used it before.
# Throttle has only two values because F-Zero has no brake button: measured
# with tools/find_buttons.py, Y/A/X/L/R all shed exactly 80 speed units a
# frame, identical to simply releasing the accelerator. A third "brake"
# value was a no-op that PPO would have wasted exploration on.
#
# That measurement also shows coasting is expensive: it loses 80/frame while
# the throttle only gains ~19/frame, so a frame off-throttle costs four to
# win back. Coasting is the brake, and it is a powerful one.
STEER_STRAIGHT, STEER_LEFT, STEER_RIGHT = 0, 1, 2
THROTTLE_COAST, THROTTLE_ACCEL = 0, 1
SHOULDER_NONE, SHOULDER_L, SHOULDER_R = 0, 1, 2

# Measured peak raw speed, used to scale the target-speed model.
TOP_SPEED_RAW = 2100

# Speed lost per frame when off the throttle, and gained per frame on it.
COAST_DECEL_PER_FRAME = 80
ACCEL_PER_FRAME = 19

ACTION_DIMS = (3, 2, 3)
NUM_ACTIONS = ACTION_DIMS[0] * ACTION_DIMS[1] * ACTION_DIMS[2]

# Not part of the action space -- bridge.lua reads this as "load savestate".
RESET_ACTION = NUM_ACTIONS


def encode_action(steer, throttle, shoulder):
    return steer * 6 + throttle * 3 + shoulder


def decode_action(code):
    return (code // 6) % 3, (code // 3) % 2, code % 3


# Escape hatch outside the action space: presses one named button on its
# own, so tools/find_buttons.py can work out what each button actually does
# rather than trusting the mapping in bridge.lua.
RAW_BUTTON_BASE = RESET_ACTION + 1
RAW_BUTTONS = ("B", "Y", "A", "X", "L", "R")

# Explicit slot operations are outside every controller action.  RESET_ACTION
# and SAVE_ACTION retain their original environment-configured behavior;
# these ranges let turn-curriculum tools select slots without relaunching an
# emulator or changing the normal action space.
SAVE_ACTION = 25
LOAD_SLOT_BASE = 32
SAVE_SLOT_BASE = 48


def _slot_action(base, slot):
    slot = int(slot)
    if not 0 <= slot <= 9:
        raise ValueError("savestate slot must be between 0 and 9")
    return base + slot


def load_slot_action(slot):
    return _slot_action(LOAD_SLOT_BASE, slot)


def save_slot_action(slot):
    return _slot_action(SAVE_SLOT_BASE, slot)


def raw_button_action(name):
    return RAW_BUTTON_BASE + RAW_BUTTONS.index(name)


ACTION_NONE = encode_action(STEER_STRAIGHT, THROTTLE_COAST, SHOULDER_NONE)
ACTION_ACCEL = encode_action(STEER_STRAIGHT, THROTTLE_ACCEL, SHOULDER_NONE)
ACTION_ACCEL_LEFT = encode_action(STEER_LEFT, THROTTLE_ACCEL, SHOULDER_NONE)
ACTION_ACCEL_RIGHT = encode_action(STEER_RIGHT, THROTTLE_ACCEL, SHOULDER_NONE)

# Checkpoints in one lap of the derived track map. Progress is no longer read
# from RAM at all -- see track_map.py and lua/bridge.lua's note on 0x000C1 --
# so this is simply the resolution tools/build_track_map.py resampled the
# recorded lap to. It is confirmed rather than assumed: replaying a recorded
# human lap through the map keeps progress monotone and in step with the
# distance actually travelled.
CHECKPOINTS_PER_LAP = 90

_TRACK_MAP = None
_TRACK_MAP_LOADED = False
_TRACK_MAP_LOCK = threading.Lock()


def _shared_track_map():
    """The track map, loaded once and shared by every client in the process.

    Held under a lock, and the "loaded" flag is set only AFTER the load
    finishes. Setting it first is a double-checked-locking bug and it was a
    live one: the genetic search drives eight environments from threads, so
    the first thread would set the flag and begin loading while every other
    thread read the flag as done and took the map while it was still None.
    Those clients then had no map, and BridgeClient._progress_from_position
    silently returned the previous value forever -- measured at 83 of 96
    episodes reporting zero progress while the cars were demonstrably driving
    at checkpoints 14 to 27.

    A missing map is fatal rather than a warning for the same reason. Progress
    that silently never advances is indistinguishable from a policy that never
    moves, and ranking a population on it is exactly the failure that wasted
    the previous ~40,000 episodes.
    """
    global _TRACK_MAP, _TRACK_MAP_LOADED
    if _TRACK_MAP_LOADED:
        return _TRACK_MAP

    with _TRACK_MAP_LOCK:
        if _TRACK_MAP_LOADED:
            return _TRACK_MAP
        from track_map import TrackMap

        try:
            loaded = TrackMap()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot load the track map ({exc}). Progress is derived from "
                "it, so without it every episode would score zero and the "
                "search would rank noise. Record a lap with lua/ram_record.lua "
                "and run tools/build_track_map.py."
            ) from exc
        _TRACK_MAP = loaded
        _TRACK_MAP_LOADED = True
        return _TRACK_MAP


def decode_bcd_progress(raw):
    """Decode the game's packed-BCD lap-timer byte.

    This decodes the LAP TIMER at 0x000C1, which used to be mistaken for track
    progress. The BCD reading itself was correct -- raw bytes really do run
    00..09, 10..19 in hexadecimal, and reading them as binary manufactured
    seven-checkpoint jumps at every decade. It was the meaning that was wrong.
    """
    value = int(raw) & 0xFF
    high, low = value >> 4, value & 0x0F
    if high <= 5 and low <= 9:
        return high * 10 + low
    # Defensive fallback for a future bridge that already publishes decoded
    # progress, or for a torn/invalid byte. The normal game path never enters
    # this branch.
    return int(raw)

# The ONLY host speed the emulator is deterministic at, and therefore the
# only one a policy may be trained or evaluated under.
#
# Above this the emulator stops being reproducible. Replaying identical inputs
# from one savestate, 30 emulated frames of left against right, and measuring
# the world-space separation of the two endpoints over 10 repeats:
#
#     speed mode      left/right span        distinct results
#     25%             15.03 +/- 0.00         1 of 10
#     50%             75.18 +/- 0.35         2 of 10
#     100%           101.41 +/- 11.39       10 of 10, range 74.24 - 116.07
#
# At 100% the same inputs from the same state produce anything from 74 to 116
# units of steering -- a 56% spread that straddles the 50% value. This is why
# an earlier 3-repeat measurement concluded 100% was "11% weaker" while a
# 10-repeat one concluded "32% stronger": both are samples from that spread,
# and neither direction is a real property. What IS real is that <= 50% is
# reproducible and 100% is not.
#
# Non-determinism is worse here than a constant scale factor would be. A
# policy can learn around consistently weaker steering; it cannot learn around
# an environment that answers differently every time, and the genetic search
# ranks genomes on differences smaller than that spread.
REQUIRED_SPEED_MODE = 50

# Escape hatch for tools that deliberately vary the host speed, such as
# tools/test_steering_speed.py, which exists precisely to measure this.
ALLOW_SPEED_MODE_MISMATCH = "FZERO_ALLOW_SPEEDMODE_MISMATCH"

POLL_SECONDS = 0.0005

# Retries for the atomic action rename, which Windows can briefly refuse
# while the Lua side has the file open.
RENAME_ATTEMPTS = 50
FRAME_TIMEOUT_SECONDS = 5.0
RESET_TIMEOUT_SECONDS = 10.0


class BridgeTimeout(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, use_mmf=True, instance=None):
        self.use_mmf = use_mmf
        self.instance = instance
        (
            self.action_file,
            self.state_file,
            self.frame_file,
            self.mmf_name,
        ) = instance_paths(instance)

        self._temp_file = f"{self.action_file}.{os.getpid()}.tmp"

        self._mmf = None
        self._sequence = 0
        self._frame_count = -1
        self._reset_count = -1
        self._lap = 0
        self._progress = 0
        self._mmf_failures = 0
        self._save_count = 0

        # Progress is derived from world position via the recorded centreline.
        # None means "no reading yet this lap", which makes the next lookup a
        # global nearest-point search instead of a windowed one.
        self._track = _shared_track_map()
        self._position = (0, 0)
        self._lap_time = 0
        self._map_index = None

        # None until the bridge publishes it. An older bridge.lua that does
        # not report its speed mode cannot be checked, so it is not blocked --
        # but it is also not silently trusted; check_speed_mode() says so.
        self._speed_mode = None

        # Assume racing until the bridge says otherwise, so an out-of-date
        # bridge.lua that does not publish the flag behaves as it always did
        # rather than reporting every frame as a menu.
        self._racing = True

    # ------------------------------------------------------------------

    def send_action(self, action):
        """Publish an action atomically.

        Writing in place truncates the file first, so bridge.lua can read it
        mid-write, fail to parse, and never see this sequence number -- while
        this side waits for a frame that will never be produced. Writing to a
        temporary file and renaming means Lua only ever sees a complete
        action, since rename is atomic on the same volume.
        """
        self._sequence += 1
        payload = f"{int(action)} {self._sequence}"

        # Windows can still hold the temp file for a moment after a rename,
        # so the OPEN can fail as well as the replace -- a training run died
        # on exactly that. Both are retried, and the temp name carries the
        # pid so parallel workers can never contend for the same one.
        for attempt in range(RENAME_ATTEMPTS):
            try:
                with open(self._temp_file, "w") as f:
                    f.write(payload)
                os.replace(self._temp_file, self.action_file)
                return
            except OSError:
                if attempt == RENAME_ATTEMPTS - 1:
                    raise
                time.sleep(POLL_SECONDS)

    def read_state(self):
        """Returns (speed, reset_count, frame_count).

        Kept to three values for callers that only need those; lap and
        progress come from read_full_state().
        """
        return self.read_full_state()[:3]

    def read_full_state(self):
        """Returns (speed, reset_count, frame_count, lap, progress).

        Tolerates a short line so an out-of-date bridge.lua still reports
        what it can rather than failing outright. A non-zero shared-memory
        failure count means the bridge has fallen back to writing frames to
        disk, so this side follows it there.
        """
        try:
            with open(self.state_file, "r") as f:
                parts = f.read().split()

            speed = int(parts[0])
            resets = int(parts[1])
            frames = int(parts[2])
            lap = int(parts[3]) if len(parts) > 3 else 0

            # parts[4] is the LAP TIMER, not progress. Kept as a diagnostic
            # only -- see decode_bcd_progress and lua/bridge.lua.
            self._lap_time = (
                decode_bcd_progress(int(parts[4])) if len(parts) > 4 else 0
            )

            # Progress is derived from the machine's world position, because
            # the game stores no lap-position counter. An older bridge.lua
            # that does not publish x/y leaves the last known value alone
            # rather than reporting a bogus zero.
            if len(parts) > 9:
                self._position = (int(parts[8]), int(parts[9]))
                progress = self._progress_from_position(*self._position)
            else:
                progress = self._progress

            if len(parts) > 10:
                self._speed_mode = int(parts[10])

            # Parsed defensively: a single unreadable trailing field must
            # not discard the whole line. One did once -- Lua wrote the text
            # "nil" for an uninitialised counter, the parse threw, this fell
            # back to cached values, and every reset timed out because the
            # reset count could never be seen to change.
            try:
                self._save_count = int(parts[6]) if len(parts) > 6 else self._save_count
            except ValueError:
                pass

            if len(parts) > 7:
                self._racing = parts[7] == "1"

            if len(parts) > 5 and int(parts[5]) > self._mmf_failures:
                self._mmf_failures = int(parts[5])
                if self.use_mmf:
                    self.use_mmf = False
                    if self._mmf is not None:
                        self._mmf.close()
                        self._mmf = None

            return speed, resets, frames, lap, progress
        except (ValueError, IndexError, OSError):
            return 0, self._reset_count, self._frame_count, self._lap, self._progress

    @property
    def speed_mode(self):
        """Host speed the bridge is running at, or None if it does not say."""
        return self._speed_mode

    def check_speed_mode(self):
        """Refuse to drive under a host speed the emulator is not exact at.

        This is the guard against training at one speed and evaluating at
        another. The difference is not a constant the policy can absorb: at
        100% the same inputs from the same savestate vary by 56% run to run
        (see REQUIRED_SPEED_MODE), so a model trained at 50% would be judged
        against an environment that no longer answers consistently.
        """
        if os.environ.get(ALLOW_SPEED_MODE_MISMATCH):
            return
        if self._speed_mode is None or self._speed_mode == REQUIRED_SPEED_MODE:
            return
        raise RuntimeError(
            f"bridge is running at host speed {self._speed_mode}%, but this "
            f"project requires {REQUIRED_SPEED_MODE}%. Above {REQUIRED_SPEED_MODE}% "
            "the emulator is not reproducible -- identical inputs from one "
            "savestate vary by 56% run to run -- so a policy trained at one "
            "speed cannot be fairly run at another. Launch emulators through "
            "tools/launch_instances.py, which sets FZERO_SPEEDMODE, or set "
            f"{ALLOW_SPEED_MODE_MISMATCH}=1 if you are deliberately measuring "
            "the difference."
        )

    def _progress_from_position(self, x, y):
        """Checkpoint index for a world position, or the timer if no map."""
        if self._track is None:
            return self._progress
        index, _wrapped = self._track.advance(x, y, self._map_index)
        self._map_index = index
        return index

    @staticmethod
    def _holds_frame(mm):
        """True if this mapping starts with a PNG, i.e. BizHawk wrote to it."""
        try:
            mm.seek(0)
            return mm.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
        except (ValueError, OSError):
            return False

    def _open_mmf(self):
        """Open BizHawk's frame mapping at the largest size that maps.

        Windows refuses a view larger than the mapping, so the largest size
        that opens is the mapping's real size -- found by binary search over
        whole pages. Mapping less than that would truncate the PNG.

        Note mmap(-1, ...) *creates* a mapping when none exists, so a
        successful open is not evidence BizHawk is publishing anything; the
        caller checks the contents for that.
        """
        low, high = 1, MMF_MAX_SIZE // MMF_PAGE_SIZE
        best = None

        while low <= high:
            mid = (low + high) // 2
            try:
                probe = mmap.mmap(-1, mid * MMF_PAGE_SIZE, tagname=self.mmf_name)
            except OSError:
                high = mid - 1
                continue

            probe.close()
            best = mid * MMF_PAGE_SIZE
            low = mid + 1

        if best is None:
            return None

        try:
            return mmap.mmap(-1, best, tagname=self.mmf_name)
        except OSError:
            return None

    def _read_frame_bytes(self):
        try:
            self._mmf.seek(0)
            data = self._mmf.read()
        except (ValueError, OSError):
            return None

        if not data.startswith(PNG_SIGNATURE):
            return None

        return data

    def read_frame(self):
        if not self.use_mmf:
            return cv2.imread(self.frame_file)

        for attempt in (0, 1):
            if self._mmf is None:
                self._mmf = self._open_mmf()
                if self._mmf is None:
                    return None

            payload = self._read_frame_bytes()
            if payload is not None:
                return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

            # The mapping we hold may predate a resize by BizHawk, so drop it
            # and probe once more before giving up.
            if attempt == 0:
                self._mmf.close()
                self._mmf = None

        return None

    # ------------------------------------------------------------------

    def advance(self, action, timeout=FRAME_TIMEOUT_SECONDS):
        """Sends an action, waits for the resulting frame, returns (speed, frame).

        Lap and progress for the same frame are available afterwards as
        .lap and .progress rather than widening the return tuple.
        """
        self.send_action(action)
        deadline = time.monotonic() + timeout

        while True:
            speed, resets, frames, lap, progress = self.read_full_state()

            if frames != self._frame_count:
                self._frame_count = frames
                self._reset_count = resets
                self._lap = lap
                self._progress = progress
                return speed, self.read_frame()

            if time.monotonic() > deadline:
                raise BridgeTimeout(
                    "bridge.lua produced no new frame. Is the Lua script "
                    "loaded in EmuHawk and not paused?"
                )

            time.sleep(POLL_SECONDS)

    def reset(self, timeout=RESET_TIMEOUT_SECONDS, slot=None):
        """Loads the savestate, returning once the bridge confirms it."""
        # bridge.lua edge-triggers the savestate load on the action CHANGING
        # to the reset code, so that holding it does not reload every frame.
        # That leaves the edge un-armed if a previous reset was abandoned
        # midway -- the bridge still has the reset code as its last action,
        # sees no change, and never reloads, so every later reset times out
        # forever. Sending one neutral frame first re-arms it.
        self.send_action(ACTION_NONE)
        try:
            self.advance(ACTION_NONE, timeout=timeout)
        except BridgeTimeout:
            pass

        _, previous_resets, _ = self.read_state()

        # Checked once per episode, which is cheap and covers every caller:
        # training, evaluation, recording and the dashboards all reset.
        self.check_speed_mode()

        # A savestate load teleports the machine, so the windowed lookup's
        # "last checkpoint" is meaningless across it. Cleared here so the
        # first reading after a reset is a global nearest-point search.
        self._map_index = None

        # Held until the bridge acknowledges, so we never hand back state
        # from before the race restarted.
        reset_action = RESET_ACTION if slot is None else load_slot_action(slot)
        self.send_action(reset_action)
        deadline = time.monotonic() + timeout

        while True:
            speed, resets, frames = self.read_state()

            if resets != previous_resets:
                self._reset_count = resets
                self._frame_count = frames
                break

            if time.monotonic() > deadline:
                self.send_action(ACTION_NONE)
                raise BridgeTimeout(
                    "bridge.lua did not acknowledge reset. Is a savestate "
                    "saved in the configured slot?"
                )

            time.sleep(POLL_SECONDS)

        return self.advance(ACTION_NONE, timeout)

    def save(self, slot, timeout=RESET_TIMEOUT_SECONDS):
        """Atomically ask BizHawk to save the current state to an explicit slot."""
        self.send_action(ACTION_NONE)
        try:
            self.advance(ACTION_NONE, timeout=timeout)
        except BridgeTimeout:
            pass

        self.read_full_state()
        previous_saves = self._save_count
        self.send_action(save_slot_action(slot))
        deadline = time.monotonic() + timeout
        while True:
            self.read_full_state()
            if self._save_count != previous_saves:
                break
            if time.monotonic() > deadline:
                self.send_action(ACTION_NONE)
                raise BridgeTimeout("bridge.lua did not acknowledge savestate save")
            time.sleep(POLL_SECONDS)
        return self.advance(ACTION_NONE, timeout)

    @property
    def save_count(self):
        """How many savestates bridge.lua has written this session."""
        return self._save_count

    @property
    def racing(self):
        """Whether the player's race is live, from the game's own state.

        False on the title screen, the course card, the results screen and
        the SELECT menu -- including the attract-mode demo, which drives at
        full speed and defeats every test based on movement or on the
        picture. See the racing_address comment in lua/bridge.lua.
        """
        return self._racing

    @property
    def lap(self):
        return self._lap

    @property
    def progress(self):
        """Checkpoint index within the current lap, 0..CHECKPOINTS_PER_LAP-1."""
        return self._progress

    @property
    def total_progress(self):
        """Monotonic progress across the whole race, in checkpoints."""
        return self._lap * CHECKPOINTS_PER_LAP + self._progress

    def close(self):
        try:
            self.send_action(ACTION_NONE)
        except OSError:
            pass

        try:
            os.remove(self._temp_file)
        except OSError:
            pass

        if self._mmf is not None:
            self._mmf.close()
            self._mmf = None
