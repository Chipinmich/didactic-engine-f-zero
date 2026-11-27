"""Can PPO learn anything here at all?

Before spending any more time designing the reward, answer the prior
question: is the machinery capable of learning on this env? Every failure so
far -- flat reward, no lap, nothing improving -- is consistent with TWO very
different causes, and they need opposite fixes:

  mechanical   the setup cannot learn regardless of the reward (too few
               policy updates, returns too large for the value head to fit,
               a wiring mistake). No amount of reward design helps.

  the reward   the machinery is fine and the reward does not rank episodes
               the way the goal does.

So: replace the reward with something trivially learnable and see if it
climbs. Here that is raw speed and nothing else -- no wall penalty, no crash
penalty, no progress term. The optimal policy is "hold the throttle", which
is one action, reachable from a random start, and unmistakable in the
numbers if it is found.

    python tools/launch_instances.py 8
    python training/control_experiment.py --timesteps 60000

If the speed reward climbs, the plumbing works and the reward shape is the
suspect. If it does NOT climb even here, the problem is mechanical and the
reward is innocent.

The A/B is worth running too, since it tests the fix rather than assuming it:

    python training/control_experiment.py --timesteps 60000 --old-settings

--old-settings restores what every previous run used: n_steps 2048 (four
policy updates in a 73k run) and unnormalised rewards.
"""

import argparse
import os
import statistics
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from env.fzero_env import DEFAULT_FRAME_SKIP, FZeroEnv, MAX_SPEED_RAW
from training.run_store import RecordingWrapper, RunWriter, load_episodes, new_run_id
from training.train import DEFAULT_FRAME_STACK, DEFAULT_N_STEPS, CLIP_REWARD

OLD_N_STEPS = 2048


class SpeedOnlyReward(gym.Wrapper):
    """Reward is how fast the machine is going, and nothing else.

    Deliberately trivial. It does not care where the machine goes, whether
    it touches a wall, or whether it finishes -- so the optimal policy is
    simply to hold the throttle, and a learner that cannot find that has a
    problem no reward can fix.

    Termination is left exactly as it was, so episodes still end on death.
    That keeps the test honest: it is the same env, only the scoring
    changes.
    """

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = info.get("speed", 0) / MAX_SPEED_RAW
        return observation, reward, terminated, truncated, info


def make_env(instance, frame_skip, run_id):
    def build():
        # steering_only used to be an opt-in FZeroEnv flag; the env no longer
        # has a throttle dimension at all (acceleration is always held), so
        # there is nothing left to pass -- the old kwarg here raised
        # TypeError on every run before this fix.
        environment = SpeedOnlyReward(
            FZeroEnv(instance=instance, frame_skip=frame_skip)
        )
        writer = RunWriter(
            run_id,
            instance=instance,
            meta={"source": "control", "note": "speed-only reward"},
        )
        return Monitor(RecordingWrapper(environment, writer))

    return build


def verdict(run_id):
    """Did it climb? Compare the first quarter of episodes with the last."""
    episodes = load_episodes(run_id)
    if len(episodes) < 20:
        print(f"\nOnly {len(episodes)} episodes -- too few to judge. Run longer.")
        return

    quarter = max(len(episodes) // 4, 5)
    first, last = episodes[:quarter], episodes[-quarter:]

    def mean(rows, key):
        return statistics.mean(row[key] for row in rows)

    # Judged on EPISODE REWARD, because that is what PPO maximises.
    # Mean speed was the wrong yardstick once wall contact started ending
    # the episode: total speed-reward then depends on surviving as well as
    # going fast, so a policy can improve markedly while its mean speed
    # falls. Measuring the thing being optimised avoids that trap.
    first_reward = mean(first, "reward")
    last_reward = mean(last, "reward")
    change = (last_reward - first_reward) / max(abs(first_reward), 1e-6) * 100

    print(f"\n{'':22s} {'first ' + str(quarter):>12s} {'last ' + str(quarter):>12s}")
    for key, label in (("mean_speed", "mean speed"), ("reward", "episode reward"),
                       ("furthest", "furthest cp"), ("steps", "steps")):
        print(f"  {label:20s} {mean(first, key):12.1f} {mean(last, key):12.1f}")

    print(f"\nepisode reward changed by {change:+.1f}% across {len(episodes)} "
          f"episodes ({first_reward:.1f} -> {last_reward:.1f})")

    # A speed-only reward is about as easy as this env gets. A few percent is
    # noise; a learner that has found "hold the throttle" shows much more.
    if change > 10:
        print("\nVERDICT: it learned. The machinery is fine, so the reward "
              "shape is the thing to look at.")
    elif change > 3:
        print("\nVERDICT: it moved, but weakly. Run longer before concluding "
              "anything -- this is within the range variance can produce here.")
    else:
        print("\nVERDICT: it did NOT learn a reward it cannot fail to learn. "
              "The problem is mechanical, not the reward design. Suspect the "
              "update count, the reward scale, or the wiring -- and note that "
              "no change to the reward function will help until this passes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- set this to "
                             "run alongside another trainer on a disjoint "
                             "range")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--old-settings", action="store_true",
                        help="n_steps 2048 and no reward normalisation, as "
                             "every previous run used")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "PPO control experiment", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    torch.set_num_threads(1)

    n_steps = OLD_N_STEPS if args.old_settings else DEFAULT_N_STEPS
    normalise = not args.old_settings
    label = "old" if args.old_settings else "fixed"
    run_id = args.run_id or new_run_id(f"control_{label}")

    builders = [make_env(i, args.frame_skip, run_id)
                for i in range(args.offset, args.offset + args.instances)]
    env = SubprocVecEnv(builders) if len(builders) > 1 else DummyVecEnv(builders)
    env = VecFrameStack(env, n_stack=args.frame_stack)

    if normalise:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=CLIP_REWARD)

    rollout = n_steps * args.instances
    print(f"Control experiment ({label} settings): speed-only reward")
    print(f"  n_steps {n_steps} x {args.instances} = {rollout} per update "
          f"-> {args.timesteps // rollout} policy updates")
    print(f"  reward normalisation: {'on' if normalise else 'off'}")
    print(f"  run '{run_id}'\n")

    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps)

    try:
        model.learn(total_timesteps=args.timesteps)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()

    verdict(run_id)


if __name__ == "__main__":
    main()
