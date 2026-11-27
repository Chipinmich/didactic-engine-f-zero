"""Tests for the run store and the behaviour-cloning dataset it feeds.

No emulator needed. The one that matters most is the stacking test: if the
clone builds its observations in a different order from VecFrameStack, the
policy learns to read its input backwards and everything downstream is
quietly wrong while every number still looks plausible.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import training.run_store as store
from training.behaviour_cloning import discounted_returns, stack_episode
from vision.road_edges import NUM_FEATURES


def test_stacking_matches_stable_baselines():
    """Our stacking must equal VecFrameStack's, frame for frame.

    Built by running the real thing: a vec env that replays a known sequence
    of observations, wrapped in the real VecFrameStack, compared against
    stack_episode over the same sequence.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    stack = 8
    steps = 20
    rng = np.random.default_rng(0)
    sequence = rng.random((steps, NUM_FEATURES)).astype(np.float32)

    class Replay(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(NUM_FEATURES,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)
            self.i = 0

        def reset(self, *, seed=None, options=None):
            self.i = 0
            observation = sequence[self.i]
            self.i += 1
            return observation, {}

        def step(self, action):
            observation = sequence[self.i]
            self.i += 1
            return observation, 0.0, False, False, {}

    env = VecFrameStack(DummyVecEnv([Replay]), n_stack=stack)

    from_sb3 = [env.reset()[0].copy()]
    for _ in range(steps - 1):
        from_sb3.append(env.step(np.array([0]))[0][0].copy())
    from_sb3 = np.stack(from_sb3)

    ours = stack_episode(sequence, stack)

    assert ours.shape == from_sb3.shape, f"{ours.shape} != {from_sb3.shape}"
    assert np.allclose(ours, from_sb3), (
        "stacking disagrees with VecFrameStack -- the newest observation "
        "must be the LAST block, with zeros before the episode started"
    )


def test_newest_observation_is_last():
    """Guards the ordering directly, so a failure says which end is wrong."""
    stack = 4
    observations = np.arange(3 * NUM_FEATURES, dtype=np.float32).reshape(3, NUM_FEATURES)
    stacked = stack_episode(observations, stack)

    # Final row: the current observation occupies the last block.
    assert np.array_equal(stacked[-1, -NUM_FEATURES:], observations[-1])
    # And the block before it is the previous observation.
    assert np.array_equal(stacked[-1, -2 * NUM_FEATURES : -NUM_FEATURES], observations[-2])
    # First row: nothing has happened yet, so everything before is zero.
    assert np.array_equal(stacked[0, -NUM_FEATURES:], observations[0])
    assert not stacked[0, :-NUM_FEATURES].any()


def test_discounted_returns_look_backwards_from_the_end():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    returns = discounted_returns(rewards, 0.5)
    assert np.allclose(returns, [1.75, 1.5, 1.0]), returns


def test_turn_quality_rewards_reducing_risk_not_merely_pressing_a_button():
    common = dict(
        off_centre=[0.3, 0.25, 0.2, 0.15],
        curve_signal=[0.4, 0.4, 0.4, 0.4],
        speed=[800, 800, 800, 800],
        progress=[30, 31, 32, 33],
        frames_in_step=[1, 1, 1, 1],
    )
    recovering = store.trajectory_quality(wall_risk=[0.7, 0.5, 0.3, 0.1], **common)
    worsening = store.trajectory_quality(wall_risk=[0.1, 0.3, 0.5, 0.7], **common)
    assert recovering["turn_recovery"] > worsening["turn_recovery"]
    assert recovering["turn_quality"] > worsening["turn_quality"]


def test_run_store_round_trip():
    """An episode written by RunWriter comes back through the readers."""
    root = tempfile.mkdtemp()
    original = store.STORE_DIR
    store.STORE_DIR = root

    try:
        writer = store.RunWriter(
            "unit_test", instance=2, meta={"algo": "test"},
            episode_fields={"generation": 7},
        )
        for step in range(5):
            writer.record(
                np.full(NUM_FEATURES, 0.5, dtype=np.float32),
                [1, 1, 0],
                2.0,
                {
                    "speed": 1000 + step,
                    "furthest": step,
                    "wall": step == 2,
                    "wall_risk": step / 10,
                    "off_centre": step / 20,
                    "curve_signal": 0.2,
                    "ray_distances": [0.1, 0.2, 0.3, 0.9, 0.7, 0.8, 0.9],
                    "racing": step != 4,
                    "applied_actions": [(2, 2), (2, 2)],
                    "reward_parts": {"progress": 10.0},
                },
            )

        summary = writer.finish_episode(
            {"furthest": 40, "start_progress": 36, "progress_gain": 4,
             "start_slot": 3, "laps_done": 0,
             "reward_parts": {"progress": 10.0}},
            terminated=True,
        )

        assert summary["steps"] == 5
        assert summary["generation"] == 7
        assert summary["start_slot"] == 3
        assert summary["start_progress"] == 36
        assert summary["progress_gain"] == 4
        assert summary["reward"] == 10.0
        assert abs(summary["wall_rate"] - 0.2) < 1e-6
        assert summary["requested_lean_rate"] == 0.0
        assert summary["actual_lean_rate"] == 1.0
        assert summary["mean_frames_per_step"] == 2.0
        assert summary["policy_agreement"] == 0.0
        assert 0.0 <= summary["turn_quality"] <= 1.0
        # The forward fan is [0.3, 0.9, 0.7], so its closest ray is 0.3.
        assert abs(summary["closest_ray_clearance"] - 0.3) < 1e-6
        assert summary["ray_balance"] > 0.0

        episodes = store.load_episodes("unit_test")
        assert len(episodes) == 1

        steps = store.load_steps("unit_test", episodes[0])
        assert steps["observations"].shape == (5, NUM_FEATURES)
        assert steps["racing"].tolist() == [True, True, True, True, False]
        assert steps["applied_actions"].tolist() == [[2, 2]] * 5
        assert steps["applied_shoulder_counts"].tolist() == [[0, 0, 2]] * 5
        assert steps["frames_in_step"].tolist() == [2] * 5
        assert np.allclose(steps["curve_signal"], [0.2] * 5)
        assert steps["agreement_frames"].tolist() == [0] * 5
        assert steps["ray_distances"].shape == (5, 7)
    finally:
        store.STORE_DIR = original
        shutil.rmtree(root, ignore_errors=True)


def test_a_half_written_line_is_skipped_not_fatal():
    """The dashboard reads this file while training is appending to it."""
    root = tempfile.mkdtemp()
    original = store.STORE_DIR
    store.STORE_DIR = root

    try:
        os.makedirs(store.run_dir("partial"))
        with open(os.path.join(store.run_dir("partial"), "episodes.0.jsonl"), "w") as fh:
            fh.write('{"episode": 0, "ended": 1.0, "reward": 5.0}\n')
            fh.write('{"episode": 1, "ende')  # caught mid-write

        episodes = store.load_episodes("partial")
        assert len(episodes) == 1, "a torn final line must be skipped, not raise"
        assert episodes[0]["reward"] == 5.0
    finally:
        store.STORE_DIR = original
        shutil.rmtree(root, ignore_errors=True)


def test_generation_summaries_round_trip_and_replace_same_generation():
    root = tempfile.mkdtemp()
    original = store.STORE_DIR
    store.STORE_DIR = root
    try:
        store.append_generation_summary(
            "lineage", {"generation": 1, "genomes": [{"genome_id": "first"}]}
        )
        store.append_generation_summary(
            "lineage", {"generation": 1, "genomes": [{"genome_id": "corrected"}]}
        )
        store.append_generation_summary(
            "lineage", {"generation": 2, "genomes": [{"genome_id": "child"}]}
        )

        generations = store.load_generations("lineage")
        assert [item["generation"] for item in generations] == [1, 2]
        assert generations[0]["genomes"][0]["genome_id"] == "corrected"
    finally:
        store.STORE_DIR = original
        shutil.rmtree(root, ignore_errors=True)


def run():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
