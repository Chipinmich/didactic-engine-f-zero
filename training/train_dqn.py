"""Train a Deep Q-Network, as an alternative to PPO.

Why it might do better here: DQN is off-policy, so every step goes into a
replay buffer and is learned from many times. This env is bottlenecked on
the emulator -- 228 emulated frames a second across eight instances, and
that is after the throughput work -- so squeezing more learning out of each
frame is exactly the right kind of trade. PPO throws its rollout away after
a few epochs on it.

Why it might do worse: the action space has to be flattened to Discrete(18)
(see env/flat_action.py), so the policy can no longer generalise across the
three dimensions, and DQN is famously more sensitive to its hyperparameters
than PPO.

    python tools/launch_instances.py 8
    python training/train_dqn.py --instances 8 --timesteps 500000

Or through the batch file:

    train_rl.bat dqn 8 500000

Episodes go to the run store like every other run, so tools/dashboard.py
shows it and it can be compared against the PPO runs directly.

NOTE: a behaviour-cloned policy cannot be loaded into this. The clone is an
actor over three dimensions; this is a Q-function over eighteen actions.
They are different networks answering different questions. Seeding DQN from
the autopilot would mean filling the replay buffer with teacher episodes
instead -- worth trying, and not done here.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

import numpy as np

from env.flat_action import FlatAction
from env.reward_scale import DEFAULT_REWARD_SCALE, RewardScale
from env.fzero_env import DEFAULT_FRAME_SKIP, WALL_CONTACT_FRAMES_LIMIT, FZeroEnv
from training.run_store import RecordingWrapper, RunWriter, new_run_id

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

DEFAULT_TIMESTEPS = 200_000
DEFAULT_FRAME_STACK = 8
TORCH_THREADS = 1
CHECKPOINT_EVERY = 20_000

# An episode here is only ~120 policy steps, so a buffer of 100k holds
# roughly 800 episodes -- plenty of history without the memory cost of the
# usual million-step default (which would also be mostly stale, since the
# policy changes fastest early on).
BUFFER_SIZE = 100_000

# Collect a few hundred episodes' worth before learning starts, so the first
# updates are not fitted to whatever the first crash happened to look like.
LEARNING_STARTS = 10_000

# Slower than the sb3 default of 0.1: exploring for a tenth of the run is
# not much when the interesting part of the track only appears once the
# machine can already get there.
EXPLORATION_FRACTION = 0.35
EXPLORATION_FINAL_EPS = 0.05


class TruthfulActionDQN(DQN):
    """DQN that records the action the game received, not the one requested.

    The safety assist can replace the policy's action before it reaches the
    emulator, measured at 14.3% of frames on a 6502-episode run. sb3 stores the
    action it emitted, so those transitions read (s, a_chosen, r, s') while r
    and s' were actually produced by a_executed.

    That is not exploration noise, which DQN handles by design -- Q(s, a) is
    off-policy precisely so it can learn from actions it would not have picked.
    It is a mislabelled target, and a systematically biased one: the assist
    overrides TOWARD safety, so a_chosen is typically the poor action (steering
    into a wall) while the good outcome came from a_executed. One transition in
    seven therefore taught the network that a bad action has good value.

    The env publishes what it actually applied as info["applied_action"], so
    the fix is to substitute it before the transition is stored. The assist
    stays switched on -- both learners should face the same environment -- and
    only the label is corrected.
    """

    def _store_transition(self, replay_buffer, buffer_action, new_obs,
                          reward, dones, infos):
        corrected = np.array(buffer_action, copy=True)
        for index, info in enumerate(infos):
            applied = (info or {}).get("applied_action")
            if applied is None:
                continue
            try:
                corrected[index] = FlatAction.flatten(applied[0], applied[1])
            except (ValueError, TypeError, IndexError):
                # A malformed info must not take down a training run; leaving
                # the original label is no worse than before this existed.
                continue
        super()._store_transition(replay_buffer, corrected, new_obs,
                                  reward, dones, infos)



def parse_start_slots(value):
    """"2,4,5,6" -> (2, 4, 5, 6); empty -> (), meaning the single default start."""
    slots = tuple(int(part) for part in str(value).split(",") if part.strip())
    if any(slot < 0 or slot > 9 for slot in slots):
        raise ValueError(f"savestate slots must be 0..9, got {value!r}")
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate savestate slots in {value!r}")
    return slots

def make_env(instance, frame_skip, run_id=None, assist_mode="full",
             reward_mode="potential", start_slots=(),
             wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT):
    def build():
        environment = FZeroEnv(instance=instance, frame_skip=frame_skip,
                               assist_mode=assist_mode,
                               reward_mode=reward_mode,
                               start_slots=start_slots,
                               wall_contact_frames=wall_contact_frames)

        if run_id is not None:
            # RecordingWrapper goes BELOW FlatAction, not above it. DQN sends
            # a flat Discrete(9) code; if RecordingWrapper sat outside
            # FlatAction it would record that raw code as the action, one
            # int instead of (steer, shoulder) -- run_store.py's record()
            # then indexes a size-1 array at position 2 and crashes.
            # Recording inside FlatAction means every trainer's run store
            # holds the same (steer, shoulder) shape, which is what
            # behaviour_cloning.py and every reader of the store expect.
            writer = RunWriter(
                run_id,
                instance=instance or 0,
                meta={"source": "dqn", "frame_skip": frame_skip},
            )
            environment = RecordingWrapper(environment, writer)

        environment = FlatAction(environment)
        # Above RecordingWrapper on purpose: the run store keeps native units
        # so dashboards and cross-run comparisons are unaffected, and only the
        # learner sees the rescaled value. See env/reward_scale.py.
        environment = RewardScale(environment, DEFAULT_REWARD_SCALE)
        return Monitor(environment)

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- set this to "
                             "run alongside another trainer on a disjoint "
                             "range, e.g. the genetic search on 8+")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--name", default="dqn_fzero")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-record", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # Varied spawns, for the same reason the genetic search uses them: a
    # policy trained from one position learns that position's corner sequence
    # and cannot be told apart from one that learned to drive. Empty keeps the
    # single default start.
    parser.add_argument("--start-slots", default="",
                        help="comma-separated savestate slots to rotate spawns over")
    parser.add_argument(
        "--wall-contact-frames", type=int, default=WALL_CONTACT_FRAMES_LIMIT,
        help="consecutive wall-contact frames required to end an episode",
    )
    parser.add_argument(
        "--reward-mode", choices=("potential", "legacy"), default="potential",
        help="potential: Ng-Harada-Russell shaping on track position, with "
             "the proxy terms it replaces dropped. legacy: the hand-weighted "
             "stack, kept for comparison.",
    )
    parser.add_argument("--assist-mode", choices=("full", "emergency"), default="full",
                        help="'full' intervenes on predicted walls, lookahead "
                             "turns and centring (default, matches earlier "
                             "runs); 'emergency' only on an actual off-road "
                             "or near-wall emergency -- matches how the "
                             "genetic policy trains, for a fair comparison "
                             "of safety-intervention rate between the two")
    args = parser.parse_args()
    if args.wall_contact_frames < 1:
        parser.error("--wall-contact-frames must be at least 1")

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "DQN training", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    torch.set_num_threads(TORCH_THREADS)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_id = None if args.no_record else (args.run_id or new_run_id(args.name))
    if run_id:
        print(f"Recording episodes to run '{run_id}'.")
        print(f"Watch it: .venv\\Scripts\\python.exe tools\\dashboard.py --run {run_id}")

    start_slots = parse_start_slots(args.start_slots)
    if start_slots:
        print(f"spawning across savestate slots {start_slots}")
    builders = [
        make_env(i, args.frame_skip, run_id, args.assist_mode,
                 reward_mode=args.reward_mode, start_slots=start_slots,
                 wall_contact_frames=args.wall_contact_frames)
        for i in range(args.offset, args.offset + max(args.instances, 1))
    ]
    env = SubprocVecEnv(builders) if len(builders) > 1 else DummyVecEnv(builders)

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)

    print(f"DQN across {len(builders)} emulator(s), "
          f"observation {env.observation_space.shape[0]} features, "
          f"{env.action_space.n} actions.")

    model_path = os.path.join(MODEL_DIR, args.name)
    replay_path = model_path + "_replay_buffer.pkl"
    resuming = args.resume and os.path.exists(model_path + ".zip")
    if resuming:
        print(f"Resuming from {model_path}.zip")
        model = TruthfulActionDQN.load(model_path, env=env)
        if os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"Restored replay experience from {replay_path}")
        else:
            print("No saved replay experience yet; resuming the learned "
                  "network with a fresh replay buffer.")
    else:
        model = TruthfulActionDQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            # One gradient step per environment, NOT per collect_rollouts call.
            #
            # sb3 counts train_freq in VECTORISED steps, so with the default
            # train_freq=4 and N parallel environments each training trigger
            # has already collected 4*N transitions -- and gradient_steps=1
            # answers it with a single update. Measured on a 7-environment run:
            # 101,920 transitions collected against 3,282 updates, one per 31.
            # Standard DQN does one per 4, so it was learning 7x less per
            # sample than the algorithm intends, purely from a default that
            # assumes a single environment.
            #
            # gradient_steps=N restores that ratio: 4*N transitions, N updates,
            # one per 4. Derived from the env count so it stays right if the
            # instance split changes -- and NOT -1, which sb3 reads as "one
            # update per transition" and would overfit the replay buffer.
            gradient_steps=max(len(builders), 1),
            tensorboard_log=LOG_DIR,
        )

    checkpoints = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // max(len(builders), 1), 1),
        save_path=MODEL_DIR,
        name_prefix=args.name,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not resuming,
            callback=checkpoints,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- saving what has been learned so far.")
    finally:
        model.save(model_path)
        model.save_replay_buffer(replay_path)
        print(f"Saved {model_path}.zip")
        print(f"Saved {replay_path}")
        env.close()


if __name__ == "__main__":
    main()
