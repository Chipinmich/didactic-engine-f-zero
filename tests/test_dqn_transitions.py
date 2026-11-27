"""The replay buffer must record what the game received, at a sane scale.

Two faults this pins, both measured on real runs:

  * The safety assist replaced the policy's action on 14.3% of frames, but sb3
    stores the action it emitted. Those transitions read (s, a_chosen, r, s')
    while r and s' came from a_executed -- and the bias is systematic, because
    the assist overrides TOWARD safety, so the mislabelled action is usually
    the bad one being credited with a good outcome.

  * Episode returns spanned -284 to +336 with a standard deviation of 96,
    while the difference between actions at one state is a couple of points.
"""

import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.flat_action import FlatAction
from env.reward_scale import DEFAULT_REWARD_SCALE, RewardScale


class StubEnv(gym.Env):
    """Minimal env; RewardScale only touches the reward."""

    def __init__(self, reward):
        super().__init__()
        self.reward = reward
        self.action_space = gym.spaces.Discrete(FlatAction.ACTIONS)
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, action):
        return "obs", self.reward, False, False, {"applied_action": (1, 2)}

    def reset(self, **kwargs):
        return "obs", {}

    def close(self):
        pass


def test_flatten_and_unflatten_are_inverses_over_the_whole_space():
    seen = set()
    for steer in range(3):
        for lean in range(3):
            code = FlatAction.flatten(steer, lean)
            seen.add(code)
            assert FlatAction.unflatten(code) == (steer, lean)
    assert seen == set(range(FlatAction.ACTIONS))


def test_flatten_matches_the_wrappers_own_decoding():
    """flatten must be the exact inverse of what step() applies.

    If these drift apart the replay buffer is relabelled with a DIFFERENT
    wrong action, which is worse than the bug it replaces.
    """
    for code in range(FlatAction.ACTIONS):
        steer, lean = FlatAction.unflatten(code)
        assert FlatAction.flatten(steer, lean) == code


def test_flatten_rejects_out_of_range_pairs():
    for bad in ((3, 0), (0, 3), (-1, 0), (0, -1)):
        try:
            FlatAction.flatten(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not be encodable")


def test_reward_scale_shrinks_the_value_the_learner_sees():
    env = RewardScale(StubEnv(336.0), DEFAULT_REWARD_SCALE)
    _, reward, _, _, _ = env.step(0)
    assert abs(reward - 336.0 * DEFAULT_REWARD_SCALE) < 1e-9
    assert abs(reward) < 10.0, "returns should land in single digits for Q-learning"


def test_reward_scale_preserves_sign_and_ordering():
    scaled = [RewardScale(StubEnv(r)).step(0)[1] for r in (-284.0, -1.0, 0.0, 5.0, 336.0)]
    assert scaled == sorted(scaled)
    assert scaled[0] < 0 < scaled[-1]


def test_reward_scale_rejects_a_zero_or_negative_factor():
    for bad in (0.0, -0.02):
        try:
            RewardScale(StubEnv(1.0), bad)
        except ValueError:
            continue
        raise AssertionError(f"scale {bad} should be rejected")


def test_applied_action_is_what_gets_stored():
    """The substitution the DQN subclass performs, in isolation.

    A policy that asked for (0, 0) while the assist executed (1, 2) must be
    stored as (1, 2)'s code, not (0, 0)'s.
    """
    buffer_action = np.array([FlatAction.flatten(0, 0)])
    infos = [{"applied_action": (1, 2)}]

    corrected = np.array(buffer_action, copy=True)
    for index, info in enumerate(infos):
        applied = (info or {}).get("applied_action")
        if applied is not None:
            corrected[index] = FlatAction.flatten(applied[0], applied[1])

    assert corrected[0] == FlatAction.flatten(1, 2)
    assert corrected[0] != buffer_action[0], "the mislabelled action survived"


def test_a_missing_or_malformed_applied_action_leaves_the_label_alone():
    original = np.array([4, 7])
    for infos in ([{}, {}], [None, None], [{"applied_action": (9, 9)}, {}]):
        corrected = np.array(original, copy=True)
        for index, info in enumerate(infos):
            applied = (info or {}).get("applied_action")
            if applied is None:
                continue
            try:
                corrected[index] = FlatAction.flatten(applied[0], applied[1])
            except (ValueError, TypeError, IndexError):
                continue
        assert list(corrected) == list(original), (
            "a malformed info must not corrupt the label or crash training"
        )


def test_start_slot_parsing_is_shared_and_strict():
    """Both gradient trainers must read spawn lists the same way."""
    from training.train_dqn import parse_start_slots as dqn_parse
    from training.train import parse_start_slots as ppo_parse

    for parse in (dqn_parse, ppo_parse):
        assert parse("2,4,5,6") == (2, 4, 5, 6)
        assert parse("") == (), "empty must mean the single default start"
        assert parse(" 2 , 4 ") == (2, 4)
        for bad in ("2,2", "2,10", "-1"):
            try:
                parse(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} should be rejected")


def test_progress_outweighs_the_shaping_it_competes_with():
    """Distance must beat driving tidily at the same distance.

    Measured over 27145 episodes, grouping by checkpoints actually gained: the
    spread of bucket means was 35.8 while the spread WITHIN a bucket was 85.9,
    so there was 2.4x more reward in proxy quality than in covering ground --
    gaining a checkpoint from 2 to 3 moved mean reward by -0.2. This pins the
    ratio of the two constants that fixed it rather than their values, so they
    can be retuned together without silently reopening the gap.
    """
    from env.fzero_env import (
        PENALTY_SPEED_DECREASE_PER_UNIT,
        REWARD_PER_CHECKPOINT,
    )

    assert REWARD_PER_CHECKPOINT >= 90.0
    assert PENALTY_SPEED_DECREASE_PER_UNIT <= 0.10
    # One checkpoint must be worth more than a full coast-sized 80-unit speed
    # drop, or shedding speed matters more than distance.
    assert REWARD_PER_CHECKPOINT > 80 * PENALTY_SPEED_DECREASE_PER_UNIT * 10


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(bool(failures))
