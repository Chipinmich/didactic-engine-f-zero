"""Train the PPO agent, optionally across several emulators at once.

Start the emulators first:

    python tools/launch_instances.py 4
    python training/train.py --instances 4

With one emulator, omit --instances and load lua/bridge.lua by hand as usual.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from env.fzero_env import DEFAULT_FRAME_SKIP, WALL_CONTACT_FRAMES_LIMIT, FZeroEnv
from training.run_store import RecordingWrapper, RunWriter, new_run_id

try:  # optional: only used for training curves
    import tensorboard  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

DEFAULT_TIMESTEPS = 200_000

# One observation is a still: nine numbers describing where the walls are
# right now. Nothing in it says which way the machine is pointing or how
# fast it is rotating, and spinning out is what ends most runs. Stacking
# consecutive observations lets the policy see that change over time.
#
# The bridge advances one emulated frame per step, so a stack of N spans
# only N/60 of a second. Eight is about an eighth of a second, which is
# enough to read a rotation without making the observation huge.
DEFAULT_FRAME_STACK = 8

# Torch defaults to a thread per core, which is tuned for large tensors.
# This policy is 72->64->64, and at that size the threading is pure
# overhead: benchmarked at the real batch, 16 threads took 1.451ms per
# update against 0.897ms on one -- 1.6x SLOWER while occupying 15 of 32
# cores that the emulators could be using.
TORCH_THREADS = 1

# Save this often (in steps) so a crashed emulator costs minutes rather
# than the whole run.
CHECKPOINT_EVERY = 20_000

# Steps each emulator collects before the policy is updated.
#
# NOT the sb3 default of 2048, and this is probably why nothing has ever
# been learned here. At 2048 across 8 emulators a rollout is 16384 steps,
# and an episode on this track is only about 124 steps -- so each update
# swallowed roughly 130 whole episodes. The longest run to date, 73k steps
# with "flat reward", therefore got FOUR policy updates. That is not a
# learning curve that failed to rise, it is a learning curve that never
# started.
#
# At 256 the same 73k steps give 35 updates and a 500k run gives 244, for
# identical emulator time -- the expensive part is driving the frames, not
# the gradient step. Each rollout still holds ~16 episodes per emulator,
# which is plenty to estimate an advantage from.
DEFAULT_N_STEPS = 256

# Episode returns here run from about -1200 to +3000, and the value head has
# to regress them with vf_coef 0.5. Unnormalised returns of that size are a
# standard way to make PPO sit still: the value loss dwarfs the policy loss
# and the useful gradient is lost in it.
#
# VecNormalize keeps a running estimate of the return's scale and divides by
# it, so the value function sees something near unit variance whatever the
# reward constants happen to be.
#
# norm_obs is deliberately FALSE. The observation is already bounded in
# [0, 1] by construction, so there is nothing to gain -- and normalising it
# would shift the input distribution out from under any behaviour-cloned
# policy, which was trained on the raw features.
NORMALISE_REWARD = True
CLIP_REWARD = 10.0


def vecnormalize_path(model_path):
    """Where a run's reward-scaling statistics live.

    Saved beside the model and reloaded with it. Skipping this quietly
    breaks --resume: the statistics restart from nothing, so the first
    updates after resuming see rewards on a completely different scale from
    the ones the policy was trained against.
    """
    return model_path + "_vecnorm.pkl"



def parse_start_slots(value):
    """"2,4,5,6" -> (2, 4, 5, 6); empty -> (), meaning the single default start."""
    slots = tuple(int(part) for part in str(value).split(",") if part.strip())
    if any(slot < 0 or slot > 9 for slot in slots):
        raise ValueError(f"savestate slots must be 0..9, got {value!r}")
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate savestate slots in {value!r}")
    return slots

def make_env(instance, frame_skip, run_id=None, assist_mode="full",
             start_slots=(), wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT):
    def build():
        environment = FZeroEnv(instance=instance, frame_skip=frame_skip,
                               assist_mode=assist_mode,
                               start_slots=start_slots,
                               wall_contact_frames=wall_contact_frames)

        if run_id is not None:
            # Built inside the worker, on purpose. Each emulator runs in its
            # own process under SubprocVecEnv and writes its own summary
            # file, because eight processes appending to one file interleave
            # and corrupt it. See training/run_store.py.
            writer = RunWriter(
                run_id,
                instance=instance or 0,
                meta={"source": "ppo", "frame_skip": frame_skip},
            )
            environment = RecordingWrapper(environment, writer)

        return Monitor(environment)

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=0,
                        help="number of parallel emulators (0 = single default bridge)")
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- match "
                             "launch_instances.py's --offset when running "
                             "alongside emulators already up elsewhere")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                        help="emulated frames each action is held for")
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK,
                        help="observations to stack; 1 disables stacking")
    parser.add_argument("--name", default="ppo_fzero", help="model filename")
    # Exposed so PPO can be run in the same environment as the other two
    # learners. It defaulted to "full" with no way to change it, while DQN
    # runs "emergency" -- comparing them would have compared environments as
    # much as algorithms.
    parser.add_argument("--assist-mode", default="full",
                        choices=("full", "emergency"),
                        help="safety assist strength, must match across a comparison")
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
    parser.add_argument("--resume", action="store_true", help="continue from the saved model")
    parser.add_argument("--run-id", default=None,
                        help="name this run in the store (default: derived from --name)")
    parser.add_argument("--no-record", action="store_true",
                        help="do not write episodes to the run store")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                        help="steps per emulator per policy update")

    parser.add_argument("--raw-reward", action="store_true",
                        help="do not normalise the reward scale (the old behaviour)")
    args = parser.parse_args()
    if args.wall_contact_frames < 1:
        parser.error("--wall-contact-frames must be at least 1")

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "PPO training", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    torch.set_num_threads(TORCH_THREADS)

    # Recorded by default: every episode from every emulator goes to the run
    # store, which is what the dashboard reads and what a later run can learn
    # from. Costs one small .npz per episode.
    run_id = None if args.no_record else (args.run_id or new_run_id(args.name))

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, args.name)

    if run_id is not None:
        print(f"Recording episodes to run '{run_id}'.")
        print(f"Watch it: .venv\\Scripts\\python.exe tools\\dashboard.py --run {run_id}")

    start_slots = parse_start_slots(args.start_slots)
    if start_slots:
        print(f"spawning across savestate slots {start_slots}")

    if args.instances > 0:
        builders = [
            make_env(i, args.frame_skip, run_id, args.assist_mode, start_slots,
                     args.wall_contact_frames)
            for i in range(args.offset, args.offset + args.instances)
        ]
        # Separate processes, since each env spends its time blocked waiting
        # on its emulator -- running them in one process would serialise
        # exactly the part that should overlap.
        env = SubprocVecEnv(builders) if args.instances > 1 else DummyVecEnv(builders)
        print(f"Training across {args.instances} emulator instance(s), "
              f"frame-skip {args.frame_skip}.")
    else:
        check_env(FZeroEnv(frame_skip=args.frame_skip))
        env = DummyVecEnv([make_env(
            None, args.frame_skip, run_id, args.assist_mode, start_slots,
            args.wall_contact_frames
        )])
        print("Training against the single default bridge.")

    # stable-baselines3 raises if given a tensorboard path without
    # tensorboard installed, so only ask for it when it is available.
    tb_log = LOG_DIR if TENSORBOARD_AVAILABLE else None
    if not TENSORBOARD_AVAILABLE:
        print("(tensorboard not installed -- training curves disabled)")

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        print(f"Stacking {args.frame_stack} observations "
              f"-> {env.observation_space.shape[0]} features.")

    # Reward scaling goes OUTSIDE the frame stack, so it only ever touches
    # the reward and the observation reaches the policy untouched.
    normalise = NORMALISE_REWARD and not args.raw_reward
    if normalise:
        stats_path = vecnormalize_path(model_path)
        if args.resume and os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = True
            print(f"Loaded reward scaling from {os.path.basename(stats_path)}")
        else:
            env = VecNormalize(
                env, norm_obs=False, norm_reward=True, clip_reward=CLIP_REWARD
            )
        print("Normalising the reward scale (--raw-reward to disable).")

    rollout = args.n_steps * max(args.instances, 1)
    print(f"n_steps {args.n_steps} x {max(args.instances, 1)} = {rollout} steps per "
          f"update, so {args.timesteps // max(rollout, 1)} policy updates in this run.")

    if args.resume and os.path.exists(model_path + ".zip"):
        print(f"Resuming from {model_path}.zip")
        model = PPO.load(model_path, env=env, tensorboard_log=tb_log,
                         n_steps=args.n_steps)
    else:
        # MlpPolicy, not CnnPolicy: the observation is a short track-feature
        # vector rather than a raw image.
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log,
                    n_steps=args.n_steps)

    checkpoints = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // max(args.instances, 1), 1),
        save_path=MODEL_DIR,
        name_prefix=args.name,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not args.resume,
            callback=checkpoints,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- saving what has been learned so far.")
    finally:
        model.save(model_path)
        # The reward-scaling statistics are part of the trained state: a
        # resume without them restarts the scale from nothing and the first
        # updates see rewards on a different scale from the ones the policy
        # learned against.
        if normalise:
            env.save(vecnormalize_path(model_path))
        print(f"Saved {model_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
