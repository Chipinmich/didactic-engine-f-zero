"""Tests for the RL reward.

Runs against a stub bridge, so no emulator is needed.

    python tests/test_reward.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import env.fzero_env as fz
from bridge_client import (
    CHECKPOINTS_PER_LAP,
    STEER_RIGHT,
    STEER_STRAIGHT,
    decode_action,
)
from env.fzero_env import FZeroEnv

# Steer straight. Throttle is no longer part of the action -- the env
# holds it down and there is no brake, so this is the whole input.
ACCELERATE = (0, 0)


class StubBridge:
    """Stands in for BridgeClient, with progress driven by the test."""

    def __init__(self, speed=1400):
        self.speed = speed
        self.lap = 0
        self.progress = 0
        self.actions = []

        # The game's own "is the race live" flag. Tests drive this directly
        # to end an episode; the env no longer infers it from the picture.
        self.racing = True

    @property
    def total_progress(self):
        return self.lap * CHECKPOINTS_PER_LAP + self.progress

    def place(self, total):
        self.lap, self.progress = divmod(max(total, 0), CHECKPOINTS_PER_LAP)

    def advance(self, action):
        self.actions.append(action)
        return self.speed, None

    def reset(self, slot=None):
        self.last_reset_slot = slot
        return self.speed, None

    def close(self):
        pass


def make_env(speed=1400, wall=False, road_pixels=600, frame_skip=fz.DEFAULT_FRAME_SKIP,
             reward_mode="potential"):
    """Env wired to a stub bridge, with the vision result stubbed too."""
    environment = FZeroEnv(frame_skip=frame_skip, reward_mode=reward_mode)
    environment.bridge = StubBridge(speed)

    state = {"wall": wall, "road_pixels": road_pixels}

    def look(spd, frame):
        return np.zeros(environment.observation_space.shape, dtype=np.float32), dict(state)

    environment._look = look
    environment._observe = lambda spd, frame: look(spd, frame)[0]
    environment.reset()
    environment.vision = state
    return environment, environment.bridge


def test_reset_start_variant_is_accelerated_deterministic_setup_not_episode_credit():
    environment, bridge = make_env(frame_skip=1)
    bridge.actions.clear()
    bridge.place(12)
    _, info = environment.reset(
        options={"start_slot": 3, "start_steers": (1, 1, 0, 0)}
    )
    decoded = [decode_action(action) for action in bridge.actions]
    assert decoded == [(1, 1, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0)]
    assert info["start_variant"] == [1, 1, 0, 0]
    assert info["start_progress"] == 12 and info["progress_gain"] == 0


def test_wall_contact_is_penalised_but_only_sustained_contact_ends_episode():
    """A one-frame mask glitch is ignored; a physical wall hit ends quickly."""
    clean, bridge_clean = make_env(wall=False, frame_skip=1)
    scraping, bridge_wall = make_env(wall=True, frame_skip=1)

    bridge_clean.place(1)
    bridge_wall.place(1)

    _, clean_reward, clean_terminated, _, _ = clean.step(ACCELERATE)
    _, wall_reward, wall_terminated, _, wall_info = scraping.step(ACCELERATE)

    assert not clean_terminated, "clean driving must not end the episode"
    assert not wall_terminated, "one bad mask frame must not end a clean run"
    assert not wall_info["wall_hit"]
    assert wall_reward < clean_reward, (
        f"a wall-touching step ({wall_reward:+.1f}) must score worse than "
        f"a clean one ({clean_reward:+.1f}) over identical progress"
    )
    assert scraping._wall_steps == 1

    while not wall_terminated:
        _, _, wall_terminated, _, wall_info = scraping.step(ACCELERATE)
    assert wall_info["wall_hit"]
    assert scraping._wall_contact_streak == fz.WALL_CONTACT_FRAMES_LIMIT


def test_experimental_one_frame_wall_mode_ends_immediately():
    environment, _ = make_env(wall=True, frame_skip=1)
    environment.wall_contact_frames = 1
    environment.terminate_on_wall = True

    _, _, terminated, _, info = environment.step(ACCELERATE)

    assert terminated
    assert info["wall_hit"]


def test_higher_speed_pays_more_every_step():
    """Speed is rewarded continuously, not only when a checkpoint ticks."""
    totals = {}
    for speed in (400, 1200, 2000):
        environment, bridge = make_env(speed=speed)
        totals[speed] = sum(environment.step(ACCELERATE)[1] for _ in range(30))

    assert totals[2000] > totals[1200] > totals[400], (
        f"more speed should pay more, got {totals}"
    )


def test_speed_decrease_gets_a_large_immediate_penalty():
    slowing, slowing_bridge = make_env(speed=1400, frame_skip=1)
    steady, _ = make_env(speed=1300, frame_skip=1)

    # Both frames finish at the same speed; only the first fell from the
    # previous measured frame, so the reward difference isolates this term.
    slowing_bridge.speed = 1300
    _, slowing_reward, _, _, info = slowing.step(ACCELERATE)
    _, steady_reward, _, _, _ = steady.step(ACCELERATE)

    expected = 100 * fz.PENALTY_SPEED_DECREASE_PER_UNIT
    assert np.isclose(steady_reward - slowing_reward, expected)
    assert info["speed_loss"] == 100
    assert info["reward_parts"]["speed_loss"] == -expected


def test_rays_reward_turning_away_and_penalise_turning_toward_close_wall():
    """LEGACY MODE ONLY. ray_response is dropped by potential shaping.

    It was the largest single penalty and the thing the agent optimised hardest
    -- 43% of all reward improvement over 4530 episodes went into it, against
    13% into progress -- while being the third count of a cost that wall
    contact already charges directly and that also ends the episode.
    """
    left_close = [0.10, 0.18, 0.30, 0.70, 0.85, 0.94, 1.0]

    def ray_env():
        environment, _ = make_env(speed=1400, frame_skip=1, reward_mode="legacy")
        environment._last_ray_distances = np.asarray(left_close, dtype=np.float32)
        environment._look = lambda speed, frame: (
            np.zeros(environment.observation_space.shape, dtype=np.float32),
            {"wall": False, "off_centre": 0.0, "wall_risk": 0.0,
             "ray_distances": left_close},
        )
        return environment

    away = ray_env()
    toward = ray_env()
    _, away_reward, _, _, away_info = away.step((2, 0))
    _, toward_reward, _, _, toward_info = toward.step((1, 0))

    assert away_reward > toward_reward
    assert away_info["ray_response_score"] == 1.0
    assert toward_info["ray_response_score"] == -1.0
    assert toward_info["ray_toward_wall_rate"] == 1.0


def test_danger_turn_reduces_speed_loss_cost_and_clearance_gain_is_recorded():
    danger = [0.10, 0.18, 0.30, 0.30, 0.80, 0.92, 1.0]
    safer = [0.18, 0.28, 0.48, 0.52, 0.82, 0.94, 1.0]
    environment, bridge = make_env(speed=1400, frame_skip=1)
    environment._last_ray_distances = np.asarray(danger, dtype=np.float32)
    environment._look = lambda speed, frame: (
        np.zeros(environment.observation_space.shape, dtype=np.float32),
        {"wall": False, "off_centre": 0.0, "wall_risk": 0.0,
         "ray_distances": safer},
    )
    bridge.speed = 1300

    _, _, _, _, info = environment.step((2, 0))

    assert np.isclose(info["speed_loss"], 100 * fz.DANGER_SPEED_LOSS_MULTIPLIER)
    assert info["mean_ray_clearance_recovery"] > 0
    assert info["reward_parts"]["ray_clearance"] > 0


def test_speed_reward_cannot_be_farmed_against_a_wall():
    """Holding speed against a wall must not out-earn standing still.

    This is why an unconditional speed bonus was removed once before. It is
    only safe because wall contact is charged every frame. This checks that
    even the recoverable contact window is a net loss and ends if sustained.
    """
    pinned, _ = make_env(speed=2000, wall=True)
    _, reward, terminated, _, _ = pinned.step(ACCELERATE)
    assert reward < 0, f"pinned at speed must be a net loss, got {reward:+.1f}"
    while not terminated:
        _, _, terminated, _, _ = pinned.step(ACCELERATE)
    assert terminated, "sustained wall contact must end the episode"


def test_speed_reward_cannot_be_farmed_by_reversing():
    """Driving flat out the wrong way must not pay."""
    environment, bridge = make_env(speed=2000)
    bridge.place(40)
    environment.step(ACCELERATE)

    total = 0.0
    for point in range(39, 29, -1):
        bridge.place(point)
        total += environment.step(ACCELERATE)[1]

    assert total < 0, f"reversing at speed must be a net loss, got {total:+.1f}"


def test_net_progress_is_signed_and_reversing_subtracts_distance():
    environment, bridge = make_env(speed=1400, frame_skip=1)
    bridge.place(10)
    environment.step(ACCELERATE)
    bridge.place(7)
    _, _, _, _, info = environment.step(ACCELERATE)

    assert info["net_progress"] == 7


def test_fixed_frame_trial_ignores_wall_finish_and_uses_exact_budget():
    environment = FZeroEnv(
        frame_skip=4,
        episode_frame_limit=5,
        terminate_on_wall=False,
        terminate_on_stall=False,
    )
    environment.bridge = StubBridge(1400)
    environment._look = lambda speed, frame: (
        np.zeros(environment.observation_space.shape, dtype=np.float32),
        {"wall": True, "road_pixels": 600},
    )
    environment._observe = lambda speed, frame: environment._look(speed, frame)[0]
    environment.reset()

    _, _, terminated, truncated, info = environment.step(ACCELERATE)
    assert not terminated and not truncated
    _, _, terminated, truncated, info = environment.step(ACCELERATE)

    assert not terminated, "walls consume trial time but do not end it"
    assert truncated and info["episode_frames"] == 5


def test_a_menu_that_looks_like_road_still_ends_the_episode():
    """The course card is flat road-grey, and must not read as racing.

    This is the bug the racing flag exists for. Between races the game shows
    "1. MUTE CITY I" over a plain grey field the same colour as the track,
    so every pixel test says wide open road ahead -- more road, in fact, than
    a real racing frame. The agent went on steering against it and training
    on the result.

    So: maximum road in view, no wall, full speed, and it must STILL end,
    on the strength of the game's own flag alone.
    """
    environment, bridge = make_env(speed=1400, wall=False, road_pixels=979)
    bridge.place(20)
    environment.step(ACCELERATE)

    bridge.racing = False

    for _ in range(fz.NOT_RACING_CONFIRM_FRAMES + 2):
        _, _, terminated, _, info = environment.step(ACCELERATE)
        if terminated:
            break

    assert terminated, "a menu must end the episode however much road it shows"
    assert info["crashed"]


def test_crashing_out_is_heavily_penalised_and_ends_the_episode():
    environment, bridge = make_env(road_pixels=600)
    bridge.place(20)
    environment.step(ACCELERATE)

    # The machine is destroyed: the game stops reporting a live race. Note
    # the road is still fully in view -- the results screen and the course
    # card both look like road, which is exactly why this is not read off
    # the picture any more.
    bridge.racing = False

    terminated = False
    worst = 0.0
    for _ in range(fz.NOT_RACING_CONFIRM_FRAMES + 2):
        _, reward, terminated, _, info = environment.step(ACCELERATE)
        worst = min(worst, reward)
        if terminated:
            break

    assert terminated, "a crash-out should end the episode"
    assert info["crashed"]
    # The step's other terms shift it slightly either way; what matters is
    # that the crash dominates everything else in the episode.
    assert worst < -fz.PENALTY_CRASH * 0.9, (
        f"crashing should cost about {fz.PENALTY_CRASH}, worst was {worst:+.1f}"
    )


def test_new_ground_pays_and_old_ground_does_not():
    """Reward is for getting further from the start, not for moving.

    Paying for the raw change in progress lets the agent rock back and
    forth over a checkpoint boundary and collect every time it re-crosses
    ground it has already covered.
    """
    environment, bridge = make_env()

    bridge.place(20)
    _, forward, _, _, _ = environment.step(ACCELERATE)
    assert forward > 0, "new ground should pay"

    # Re-cover ground already driven, repeatedly.
    total = 0.0
    for _ in range(5):
        for point in (14, 17, 20):
            bridge.place(point)
            _, reward, _, _, _ = environment.step(ACCELERATE)
            total += reward

    assert total < 0, f"re-covering old ground must not pay, got {total:+.1f}"
    # The stub spawns at checkpoint 0, so distance and checkpoint coincide
    # here; test_distance_is_measured_from_the_spawn covers the difference.
    assert environment._furthest_distance == 20

    bridge.place(24)
    _, reward, _, _, _ = environment.step(ACCELERATE)
    assert reward > 0, "genuinely new ground should still pay"


def test_savestate_reload_is_not_a_windfall_or_a_crash():
    """A reset jumps progress wildly; it must count as neither gain nor loss."""
    environment, bridge = make_env()
    bridge.place(60)
    environment.step(ACCELERATE)

    # Savestate reload: progress snaps back to the start line.
    bridge.place(0)
    _, reward, _, _, _ = environment.step(ACCELERATE)

    assert reward > -fz.PENALTY_PER_CHECKPOINT_LOST, (
        f"a reload must not be charged as lost ground, got {reward:+.1f}"
    )


def test_driving_then_hitting_a_wall_beats_sitting_still():
    """Getting somewhere, even if it ends in a wall, must beat never moving.

    A machine that makes real progress and then clips a wall must score
    better than one that sat still for the same length of time. If a large
    enough wall penalty ever made that false, standing still would be the
    optimal strategy -- the trap env/fzero_env.py documents twice.

    This deliberately asserts nothing about whether the clip TERMINATES.
    It used to, incidentally, because the contact limit was 3 frames and a
    single held action is 4 -- so any touch ended the run. The limit is now
    45 frames and a brief clip is survivable, which is the point of the
    change. Termination itself is covered by the test below; mixing the two
    concerns here is what made this test fail for a reason unrelated to what
    it exists to protect.
    """
    STEPS = 20

    driving, bridge = make_env(speed=1100, wall=False)
    drove = 0.0
    for step in range(STEPS):
        driving.vision["wall"] = step == STEPS - 1  # clip a wall on the last step
        bridge.place(step + 1)
        _, reward, terminated, _, _ = driving.step(ACCELERATE)
        drove += reward
        if terminated:
            break

    idle, _ = make_env(speed=0, wall=False)
    idled = sum(idle.step(ACCELERATE)[1] for _ in range(STEPS))

    assert drove > idled, (
        f"driving then clipping a wall ({drove:+.0f}) must beat idling for "
        f"the same length of time ({idled:+.0f}), or the wall penalty has "
        "made standing still optimal"
    )


def test_brief_contact_survives_but_sustained_contact_ends_the_run():
    """The contact limit must forgive a scrape and still stop a grind.

    Guards both halves of the threshold, because only testing one is how
    this regressed before: a limit strict enough to catch a grind was also
    catching every ordinary brush, which halved reachable distance on an
    unchanged driver (median checkpoint 70 -> 36 over 200 autopilot
    episodes) and made a lap unreachable in 8700 genetic-search episodes.

    frame_skip=1 so one step is one frame and the streak is countable
    exactly. The threshold is set on the instance rather than the module
    constant, so this tests the mechanism at whatever value ships.
    """
    LIMIT = 5

    environment, bridge = make_env(wall=False, frame_skip=1)
    environment.wall_contact_frames = LIMIT

    # A scrape shorter than the limit must NOT end the run.
    environment.vision["wall"] = True
    for frame in range(LIMIT - 1):
        _, _, terminated, _, info = environment.step(ACCELERATE)
        assert not terminated, (
            f"{frame + 1} contact frames ended the run, but the limit is {LIMIT}"
        )
    assert environment._wall_contact_streak == LIMIT - 1

    # One clean frame must reset it, or a machine that brushes a wall once
    # every few frames would accumulate its way to a false hit.
    environment.vision["wall"] = False
    environment.step(ACCELERATE)
    assert environment._wall_contact_streak == 0, "a clean frame must reset the streak"

    # Unbroken contact reaching the limit must still end the run.
    environment.vision["wall"] = True
    terminated = False
    for _ in range(LIMIT):
        _, _, terminated, _, info = environment.step(ACCELERATE)
        if terminated:
            break
    assert terminated, f"{LIMIT} unbroken contact frames must end the run"
    assert info["wall_hit"]


def test_contact_termination_can_be_disabled():
    """Zero means the game's own rules decide, not an artificial limit.

    With contact-termination off, only the no-progress timer and the game's
    destruction flag end an episode -- which is how episodes ended before
    this rule existed, and what makes a full lap reachable again.
    """
    environment, bridge = make_env(wall=True, frame_skip=1)
    environment.wall_contact_frames = 0

    for step in range(40):
        bridge.place(step + 1)  # keep progressing so the no-progress timer stays clear
        _, _, terminated, _, info = environment.step(ACCELERATE)
        assert not terminated, "contact must not end the run when the limit is 0"
    assert not info["wall_hit"]
    # The per-frame penalty must still be charged -- disabling termination
    # must not quietly make wall contact free.
    assert environment._reward_parts["wall"] < 0


def test_a_stopped_machine_ends_its_episode():
    """The machine must keep moving -- it cannot park against a wall.

    Acceleration is always held, so being slow means scrubbing something.
    In time-trial mode wall contact does NOT end the episode, so without
    this a wedged genome grinds out the whole trial for free. Measured on a
    live generation before this existed: mean wall contact 54.8% of frames
    at a mean speed of 373 out of a 2100 top speed.
    """
    environment, bridge = make_env(speed=100, wall=False, frame_skip=1)
    environment.min_speed_floor = 400
    environment.min_speed_frames = 5

    terminated = False
    for frame in range(4):
        bridge.place(frame + 1)
        _, _, terminated, _, info = environment.step(ACCELERATE)
        assert not terminated, f"ended after {frame + 1} slow frames, limit is 5"

    bridge.place(10)
    _, _, terminated, _, info = environment.step(ACCELERATE)
    assert terminated, "5 consecutive frames below the floor must end the episode"
    assert info["stalled_out"]


def test_speed_above_the_floor_resets_the_stall_counter():
    """A momentary scrub must not accumulate into a false stall."""
    environment, bridge = make_env(speed=100, wall=False, frame_skip=1)
    environment.min_speed_floor = 400
    environment.min_speed_frames = 5

    for _ in range(3):
        environment.step(ACCELERATE)
    assert environment._slow_frames == 3

    bridge.speed = 900          # back up to a real driving speed
    environment.step(ACCELERATE)
    assert environment._slow_frames == 0, "recovering speed must reset the counter"


def test_the_speed_floor_can_be_disabled():
    """0 means the floor is off, for callers that want the old behaviour."""
    environment, bridge = make_env(speed=50, wall=False, frame_skip=1)
    environment.min_speed_floor = 0

    for frame in range(30):
        bridge.place(frame + 1)
        _, _, terminated, _, info = environment.step(ACCELERATE)
        assert not terminated, "a disabled floor must never end the episode"
    assert not info["stalled_out"]


def test_offroad_is_visible_in_the_observation():
    """Leaving the track must be distinguishable from being boxed in.

    Reading only the wall gaps, a machine wedged between two close walls,
    one off-track with the road to its left, and one off-track with the road
    to its RIGHT all produce the same vector of zeros -- and two of those
    need opposite actions, so the agent could not learn to recover.
    """
    from vision.road_edges import features_from_readings

    def rows_with(on_road=True, nearest=None, left=0, right=0):
        return [
            {
                "left_px": left,
                "right_px": right,
                "on_road": on_road,
                "nearest_road_x": nearest,
                "center_x": 128,
            }
            for _ in range(4)
        ]

    wedged = features_from_readings(rows_with(), 0.5)
    road_left = features_from_readings(rows_with(on_road=False, nearest=20), 0.5)
    road_right = features_from_readings(rows_with(on_road=False, nearest=236), 0.5)

    assert not np.array_equal(wedged, road_left)
    assert not np.array_equal(wedged, road_right)
    assert not np.array_equal(road_left, road_right), (
        "road to the left and road to the right must not look identical"
    )


def test_power_is_not_rewarded():
    """Power must never appear in the reward.

    It is a consequence of hitting walls rather than an achievement, and
    paying for a full bar would reward a machine that crawls round
    untouched. The damage is penalised instead.
    """
    import inspect

    source = inspect.getsource(fz)
    reward_section = source.split("def step(")[1]
    for term in ("read_power", "PowerTracker", "power_level", "hud"):
        assert term not in reward_section, (
            f"{term} appears in the reward path -- power must not be scored"
        )

    assert "power" not in fz.REWARD_PARTS


def test_standing_still_loses_reward():
    """Doing nothing must cost, or cruising is as good as racing."""
    environment, bridge = make_env(speed=0)
    total = sum(environment.step(ACCELERATE)[1] for _ in range(20))
    assert total < 0, f"no progress should be a net loss, got {total:+.1f}"


def test_dawdling_loses_reward():
    """Making progress slowly must still be worse than not dawdling.

    Speed is scored against a baseline for this reason: a flat per-step
    bonus pays more the longer you take.
    """
    environment, bridge = make_env(speed=600)
    total = 0.0
    for point in range(1, 11):
        bridge.place(point)
        for _ in range(20):
            total += environment.step(ACCELERATE)[1]

    fast, bridge_fast = make_env(speed=2000)
    quick = 0.0
    for point in range(1, 11):
        bridge_fast.place(point)
        for _ in range(20):
            quick += fast.step(ACCELERATE)[1]

    assert quick > total, (
        f"covering the same ground faster must pay more: {quick:+.1f} vs {total:+.1f}"
    )


def test_faster_lap_pays_more():
    """A lap completed in fewer steps must be worth strictly more."""
    rewards = {}

    for label, steps_per_checkpoint in (("fast", 1), ("medium", 3), ("slow", 8)):
        environment, bridge = make_env()
        total = 0.0
        point = 0

        while True:
            point += 1
            bridge.place(point)
            for _ in range(steps_per_checkpoint):
                _, reward, terminated, _, info = environment.step(ACCELERATE)
                total += reward
                if terminated:
                    break
            if terminated:
                break

        assert info["finished"], f"{label} lap should have finished"
        rewards[label] = total

    assert rewards["fast"] > rewards["medium"] > rewards["slow"], (
        f"faster laps must pay more, got {rewards}"
    )


def test_lap_completion_dominates():
    """Finishing must beat giving up, whatever the step cost."""
    environment, bridge = make_env()

    give_up = sum(
        environment.step(ACCELERATE)[1] for _ in range(fz.NO_PROGRESS_FRAMES_LIMIT)
    )

    environment, bridge = make_env()
    total = 0.0
    for point in range(1, CHECKPOINTS_PER_LAP + 1):
        bridge.place(point)
        for _ in range(4):
            _, reward, terminated, _, _ = environment.step(ACCELERATE)
            total += reward
            if terminated:
                break
        if terminated:
            break

    assert total > give_up, (
        f"finishing ({total:+.1f}) must beat giving up ({give_up:+.1f})"
    )


def _readings(left, right):
    """Scan rows where only the near row (last) carries the given gaps."""
    far = {"left_px": 40, "right_px": 40, "on_road": True, "center_x": 128}
    near = {"left_px": left, "right_px": right, "on_road": True, "center_x": 128}
    return [dict(far), dict(far), dict(far), near]


def test_off_centre_measures_position_not_road_width():
    """Centred is zero on any width; hard against a wall is one."""
    assert fz.off_centre(_readings(50, 50)) == 0.0
    assert fz.off_centre(_readings(10, 10)) == 0.0, "narrow road, still centred"
    assert fz.off_centre(_readings(0, 80)) == 1.0, "against the left wall"
    assert fz.off_centre(_readings(80, 0)) == 1.0, "against the right wall"

    drifting = fz.off_centre(_readings(30, 70))
    assert 0.0 < drifting < 1.0, drifting

    # Only one wall in frame -- the common case on the near row, where the
    # road is widest and both walls fall outside a 256px screen. There is no
    # midpoint to measure against, so it falls back to clearance from the
    # wall it CAN see. A first version returned 0.0 here and was therefore
    # dead on most real frames.
    assert fz.off_centre(_readings(None, 200)) == 0.0, "miles clear, fine"
    assert fz.off_centre(_readings(None, 0)) == 1.0, "against the visible wall"
    hugging = fz.off_centre(_readings(None, 15))
    assert 0.0 < hugging < 1.0, hugging
    assert fz.off_centre(_readings(15, None)) == hugging, "symmetric"

    assert fz.off_centre([]) == 0.0


def test_staying_centred_pays_more_than_drifting():
    """LEGACY MODE ONLY. The centre term is dropped by potential shaping,
    which signals the same thing (a good line gets further) without a
    hand-weighted proxy that can be optimised in preference to the goal.
    """
    totals = {}
    for label, off in (("centred", 0.0), ("drifting", 0.5), ("hugging", 1.0)):
        environment, bridge = make_env(reward_mode="legacy")
        environment.vision["off_centre"] = off
        totals[label] = sum(environment.step(ACCELERATE)[1] for _ in range(20))

    assert totals["centred"] > totals["drifting"] > totals["hugging"], (
        f"staying centred must pay best, got {totals}"
    )


def test_centring_is_a_penalty_not_a_bonus():
    """LEGACY MODE ONLY -- but the trap it guards still matters.

    A centring BONUS is farmable: any positive per-step payout is banked more
    times by a slow lap than a fast one. Potential shaping avoids the trap
    differently, by telescoping, so it can afford to drop the term entirely.

    Perfectly centred must be worth exactly zero from this term, not a
    payment: a positive per-step payout is banked once per step, so a slow lap
    collects it more times than a fast one.
    """
    centred, _ = make_env(reward_mode="legacy")
    centred.vision["off_centre"] = 0.0
    for _ in range(10):
        centred.step(ACCELERATE)

    assert centred._reward_parts["centre"] == 0.0, (
        "perfect centring must cost nothing and pay nothing, got "
        f"{centred._reward_parts['centre']:+.3f}"
    )

    drifting, _ = make_env(reward_mode="legacy")
    drifting.vision["off_centre"] = 0.6
    for _ in range(10):
        drifting.step(ACCELERATE)

    assert drifting._reward_parts["centre"] < 0.0, "being off-centre must cost"


def test_no_coast_but_lean_controls_remain():
    """Acceleration is fixed; lean-left and lean-right remain selectable."""
    environment = FZeroEnv()
    environment.bridge = StubBridge(1400)

    assert tuple(environment.action_space.nvec) == (3, 3), (
        "steer left/straight/right x lean none/left/right"
    )

    sent = []
    environment.bridge.advance = lambda code: (sent.append(code), (1400, None))[1]

    environment._look = lambda spd, frame: (
        np.zeros(environment.observation_space.shape, dtype=np.float32),
        {"wall": False, "road_pixels": 600, "off_centre": 0.0},
    )
    environment._observe = lambda spd, frame: environment._look(spd, frame)[0]
    environment.reset()

    sent.clear()
    for steer in (0, 1, 2):
        environment.step((steer, steer))

    from bridge_client import decode_action

    assert sent, "nothing was sent to the bridge"
    for code in sent:
        _, throttle, shoulder = decode_action(code)
        assert throttle == 1, "throttle must be held down -- coasting is a brake"
    steers = {decode_action(code)[0] for code in sent}
    shoulders = {decode_action(code)[2] for code in sent}
    assert steers == {0, 1, 2}, f"all three steering inputs must reach the bridge, got {steers}"
    assert shoulders == {0, 1, 2}, f"all lean inputs must reach the bridge, got {shoulders}"


def test_curve_shortens_the_hold_but_still_steers_on_every_held_frame():
    """A curve shortens the hold; it no longer costs the whole frame skip.

    The assist re-evaluates inside the hold, so holding an action for longer
    does not mean stale steering -- every emulated frame of the step still
    gets the corrected direction and lean. That matters because a one-frame
    hold was firing on 91.9% of decisions, which made the pad flicker rather
    than hold a direction long enough to turn the machine.
    """
    environment, bridge = make_env()
    curve = [
        {"left_px": 15, "right_px": 75, "on_road": True},
        {"left_px": 20, "right_px": 70, "on_road": True},
        {"left_px": 60, "right_px": 60, "on_road": True},
        {"left_px": 60, "right_px": 60, "on_road": True},
    ]
    environment._assist.observe(curve)
    sent = []
    bridge.advance = lambda code: (sent.append(code), (bridge.speed, None))[1]

    _, _, _, _, info = environment.step((0, 0))

    from bridge_client import SHOULDER_R, STEER_RIGHT, THROTTLE_ACCEL, decode_action

    assert len(sent) == 3, f"a curve should shorten the hold to 3, sent {len(sent)}"
    for code in sent:
        assert decode_action(code) == (STEER_RIGHT, THROTTLE_ACCEL, SHOULDER_R), (
            "every frame of the hold must carry the corrected action"
        )
    assert info["frames_in_step"] == 3
    assert info["applied_action"] == (STEER_RIGHT, SHOULDER_R)


def test_step_exposes_seven_rays_to_episode_recording():
    """The policy's ray observation must also reach storage/dashboard info."""
    environment, _ = make_env(frame_skip=1)
    expected = [0.1, 0.2, 0.3, 0.9, 0.7, 0.8, 1.0]
    environment.vision["ray_distances"] = expected

    _, _, _, _, info = environment.step(ACCELERATE)

    assert info["ray_distances"] == expected


def test_turn_start_slots_rotate_and_progress_is_measured_from_each_start():
    environment = FZeroEnv(frame_skip=1, assist_mode="emergency", start_slots=(2, 3))
    bridge = StubBridge()
    environment.bridge = bridge
    environment._look = lambda speed, frame: (
        np.zeros(environment.observation_space.shape, dtype=np.float32),
        {"wall": False, "road_pixels": 600, "off_centre": 0.0},
    )
    environment._observe = lambda speed, frame: environment._look(speed, frame)[0]

    bridge.place(28)
    _, first = environment.reset()
    assert bridge.last_reset_slot == 2
    assert first["start_progress"] == 28 and first["progress_gain"] == 0

    bridge.place(36)
    _, second = environment.reset()
    assert bridge.last_reset_slot == 3
    bridge.place(39)
    info = environment._info(bridge.speed)
    environment._furthest_distance = 3
    info = environment._info(bridge.speed)
    assert info["progress_gain"] == 3


def test_a_turn_is_held_long_enough_to_actually_move_the_machine():
    """Steering for one or two frames barely moves anything.

    Measured by holding one direction from a fixed savestate against holding
    straight, the machine is pulled off line by 35 world units after 5 frames
    and 90 after 10, while travelling 567 forward over 59. The assist used to
    hand control back after a single frame on 91.9% of decisions, so a turn
    never lasted long enough to bite.
    """
    environment, bridge = make_env()
    # Clear road: nothing for the assist to react to.
    environment._assist.observe([
        {"left_px": 60, "right_px": 60, "on_road": True} for _ in range(4)
    ])
    sent = []
    bridge.advance = lambda code: (sent.append(code), (bridge.speed, None))[1]

    _, _, _, _, info = environment.step((STEER_RIGHT, 0))

    assert info["frames_in_step"] >= environment.turn_commitment, (
        f"a turn held for only {info['frames_in_step']} frames"
    )
    assert len(sent) == info["frames_in_step"]


def test_going_straight_is_not_extended_by_the_turn_commitment():
    environment, bridge = make_env()
    environment._assist.observe([
        {"left_px": 60, "right_px": 60, "on_road": True} for _ in range(4)
    ])
    bridge.advance = lambda code: (bridge.speed, None)

    _, _, _, _, info = environment.step((STEER_STRAIGHT, 0))

    assert info["frames_in_step"] == environment.frame_skip


def test_an_emergency_still_cuts_a_committed_turn_short():
    """The commitment must not drive the machine into a wall it can see."""
    environment, bridge = make_env()
    environment._assist.observe(
        [{"left_px": None, "right_px": None, "on_road": False,
          "nearest_road_x": 10, "center_x": 128} for _ in range(4)]
    )
    assert environment._assist.state["emergency"]
    bridge.advance = lambda code: (bridge.speed, None)

    _, _, _, _, info = environment.step((STEER_RIGHT, 0))

    assert info["frames_in_step"] == 1


def test_potential_shaping_telescopes_so_ground_cannot_be_recrossed_for_profit():
    """Rocking back and forth over a checkpoint must be worth exactly zero.

    This is the property that lets the shaping be dense without being
    farmable: consecutive terms cancel, so only where you END matters, never
    the path taken. With gamma < 1 there is a small residual drift, so this
    checks the sum is negligible rather than exactly zero.
    """
    environment, bridge = make_env(frame_skip=1)
    bridge.place(10)
    environment.step(ACCELERATE)
    environment._reward_parts["potential"] = 0.0

    for _ in range(6):
        bridge.place(11)
        environment.step(ACCELERATE)
        bridge.place(10)
        environment.step(ACCELERATE)

    recrossed = environment._reward_parts["potential"]

    # The same twelve steps spent genuinely advancing, for comparison.
    forward, forward_bridge = make_env(frame_skip=1)
    forward_bridge.place(10)
    forward.step(ACCELERATE)
    forward._reward_parts["potential"] = 0.0
    for step in range(12):
        forward_bridge.place(11 + step)
        forward.step(ACCELERATE)
    advanced = forward._reward_parts["potential"]

    assert recrossed < 0 < advanced, (
        f"re-crossing banked {recrossed:+.2f}, advancing {advanced:+.2f}"
    )
    assert advanced > 5 * abs(recrossed), (
        "re-crossing the same ground must be worth nothing next to real "
        f"progress: {recrossed:+.2f} against {advanced:+.2f}"
    )


def test_ending_an_episode_forfeits_the_potential_that_was_built_up():
    """The terminal cost the legacy reward was missing.

    99.8% of DQN episodes ended on a wall hit, and a wall hit charged nothing:
    PENALTY_CRASH fires on `crashed`, a different flag, which was False on all
    4530 of them. Continuing cost -5.53 a step, so ending was the best move
    available. PHI(terminal) = 0 makes ending cost the ground gained, derived
    rather than hand-tuned.
    """
    # Build the potential up the way an episode actually does, over several
    # clean frames -- self._potential holds PHI at the START of a frame, so a
    # run that ends on its very first frame has correctly banked nothing yet.
    environment, bridge = make_env(wall=False, frame_skip=1)
    for position in range(1, 16):
        bridge.place(position)
        environment.step(ACCELERATE)
    assert environment._potential > 0.0, "no potential was built"
    environment._reward_parts["potential"] = 0.0

    environment.vision["wall"] = True
    environment.wall_contact_frames = 1
    _, _, terminated, _, info = environment.step(ACCELERATE)

    assert terminated and info["wall_hit"]
    assert info["reward_parts"]["potential"] < 0.0, (
        "ending an episode must forfeit the potential, not be free"
    )


def test_a_lap_still_beats_giving_up_despite_forfeiting_its_potential():
    """POTENTIAL_SCALE is bounded by the lap bonus, and this is the bound.

    Finishing zeroes the potential like any other termination, so the lap must
    pay more than it gives up. At POTENTIAL_SCALE 12 it did not -- completing
    scored -3727 against -33 for crashing immediately.
    """
    # The full lap length, not the ~82 a slot-2 start covers: the bound has
    # to hold for any start position, not only the ones in use today.
    lap = fz.POTENTIAL_SCALE * CHECKPOINTS_PER_LAP
    assert lap < fz.REWARD_LAP_COMPLETE, (
        f"a lap builds {lap:.0f} of potential but only pays "
        f"{fz.REWARD_LAP_COMPLETE:.0f} at the slowest bonus"
    )


def test_potential_mode_drops_the_proxies_it_replaces():
    environment, bridge = make_env(frame_skip=1)
    environment.vision["off_centre"] = 1.0
    for _ in range(5):
        environment.step(ACCELERATE)

    for dropped in fz.POTENTIAL_MODE_DROPS:
        assert environment._reward_parts[dropped] == 0.0, (
            f"{dropped} must not reach the learner in potential mode"
        )


def test_legacy_mode_still_charges_them():
    """The comparison arm has to actually differ, or the A/B means nothing."""
    environment, bridge = make_env(frame_skip=1, reward_mode="legacy")
    environment.vision["off_centre"] = 1.0
    for _ in range(5):
        environment.step(ACCELERATE)

    assert environment._reward_parts["centre"] < 0.0
    assert environment._reward_parts["step_cost"] < 0.0
    assert environment._reward_parts["potential"] == 0.0


def test_a_teleport_neither_credits_nor_charges_the_shaping():
    """A savestate reload is not progress and not a loss.

    Without baselining the potential across the jump it landed as a single
    -360 shaping term, because PHI is relative to the episode start and the
    start did not move with the teleport.
    """
    environment, bridge = make_env(frame_skip=1)
    bridge.place(10)
    environment.step(ACCELERATE)
    environment._reward_parts["potential"] = 0.0

    bridge.place(10 + CHECKPOINTS_PER_LAP // 2 + 5)
    _, reward, _, _, info = environment.step(ACCELERATE)

    assert abs(info["reward_parts"]["potential"]) < fz.POTENTIAL_SCALE, (
        f"a teleport moved the potential by "
        f"{info['reward_parts']['potential']:+.1f}"
    )


def test_distance_is_measured_from_the_spawn_not_the_checkpoint_number():
    """Every spawn must be the same task.

    Reporting the raw checkpoint number made a late spawn look like instant
    progress and, worse, made a "lap" cheaper the further along it started:
    measured against a recorded human lap, episodes flagged finished had
    covered 60% of a lap from checkpoint 9 but only 33% from checkpoint 41.
    """
    environment, bridge = make_env(frame_skip=1)
    bridge.place(40)
    environment.reset()

    bridge.place(52)
    _, _, _, _, info = environment.step(ACCELERATE)

    assert info["progress_gain"] == 12, (
        f"driving 12 checkpoints from a spawn at 40 must read 12, got "
        f"{info['progress_gain']}"
    )
    assert info["start_progress"] == 40, "the spawn is still reported, for diagnosis"


def test_crossing_the_line_keeps_counting_instead_of_jumping():
    """The modulo wrap is real motion and must not corrupt the total.

    Position is a modulo-90 counter, so the line reads as a -89 step. Treating
    that as a teleport and shifting the episode baseline made progress_gain
    leap ~89 in one frame and start_progress go negative (-52, -81 measured on
    a live PPO run), then freeze because the frontier could never be beaten.
    """
    environment, bridge = make_env(frame_skip=1)
    bridge.place(80)
    environment.reset()

    for point in (85, 88, 2, 6):          # 88 -> 2 crosses the line
        bridge.place(point)
        _, _, _, _, info = environment.step(ACCELERATE)

    # 80 -> 96 in real distance, which is 6 past the line.
    assert info["progress_gain"] == 16, f"got {info['progress_gain']}"
    assert info["start_progress"] == 80, "the baseline must not move"
    assert info["laps_done"] == 0, "16 checkpoints is not a lap"


def test_a_lap_is_ninety_checkpoints_driven_wherever_it_started():
    environment, bridge = make_env(frame_skip=1)
    bridge.place(41)
    environment.reset()

    # Ten checkpoints a frame: brisk, but inside what track_map's windowed
    # search can actually report, so the wrap logic sees realistic steps.
    position = 41
    for _ in range(10):
        position = (position + 10) % CHECKPOINTS_PER_LAP
        bridge.place(position)
        _, _, terminated, _, info = environment.step(ACCELERATE)
        if terminated:
            break

    assert info["progress_gain"] >= CHECKPOINTS_PER_LAP
    assert info["laps_done"] == 1, (
        f"a lap is {CHECKPOINTS_PER_LAP} checkpoints driven, got "
        f"laps_done {info['laps_done']} for {info['progress_gain']} checkpoints"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
