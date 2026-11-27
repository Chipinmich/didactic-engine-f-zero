"""Train the PPO policy on the autopilot's driving, before PPO ever runs.

Hand-rolled, because `imitation` cannot be installed here -- it pins
gymnasium back to 0.29 and breaks stable-baselines3. There is not much to it
anyway: the policy already defines a distribution over actions, so cloning
is just maximising the log-probability of the teacher's action under it.

    python training/record_teacher.py --instances 8 --episodes 40
    python training/behaviour_cloning.py --run teacher_20260820_193000
    train_rl.bat 8 500000 --resume --name bc_ppo_fzero

The output is an ordinary PPO save, so training continues from it with
--resume and nothing downstream needs to know this happened.

Two details that are easy to get wrong and silently ruin it:

  stacking   the agent trains behind VecFrameStack, so it never sees a bare
             observation -- it sees 8 of them (vision.road_edges.NUM_FEATURES
             wide each), oldest first, newest last, zero-padded at the start
             of an episode. The clone has to be built the same way round or
             the policy learns to read its input backwards. Checked against
             stable_baselines3.common.vec_env.stacked_observations, which
             rolls the buffer and writes the newest frame at the end.

  the value  network is NOT cloned by the policy loss -- nothing in the
             teacher's actions constrains it. Left random, the first PPO
             updates compute advantages against noise and can undo the
             clone before it is worth anything, so it is fitted here to the
             discounted returns actually recorded.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from training.run_store import load_episodes, load_steps
from vision.road_edges import NUM_FEATURES

# steer, shoulder -- matches FZeroEnv.action_space (MultiDiscrete([3, 3])).
# NOT bridge_client.ACTION_DIMS (3, 2, 3): that still includes a throttle
# slot from when the game genuinely had one. FZeroEnv now holds the
# throttle down internally and its action space has no such dimension, so
# building this env's spec from the old constant silently created a
# 3-dimensional policy that could never match what record_teacher.py now
# records (2 columns: steer, shoulder).
ACTION_DIMS = (3, 3)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

DEFAULT_FRAME_STACK = 8
DEFAULT_BATCH = 256

# Higher than PPO's own 3e-4, and it matters. stable-baselines3 initialises
# the action head orthogonally at gain 0.01, so every action starts out
# almost equally likely and the gradients through that layer are tiny --
# tuned for a policy that should explore before it commits, which is the
# opposite of what cloning wants.
#
# Measured on a synthetic teacher with a rule that is trivially learnable
# (steer toward the side with more room): at 3e-4 the clone was still at
# 66.7% agreement after 150 epochs and flat-looking; at 1e-3 it reaches
# 95.8% in 60. The lower rate was not converging slowly so much as looking
# exactly like a clone that cannot learn.
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 30

# Matches PPO's own default, so the value head is fitted on the same
# quantity PPO will go on to regress against.
DEFAULT_GAMMA = 0.99

# Held back to report honestly. Cloning accuracy on data the fit has already
# seen says nothing -- this policy has enough parameters to memorise a small
# run, and "97% agreement" on the training split would mean nothing at all.
VALIDATION_FRACTION = 0.1


class _SpecEnv(gym.Env):
    """Carries the spaces so PPO can be constructed without an emulator.

    PPO needs an env to build its networks from. Cloning is entirely offline,
    so this supplies the shapes and nothing else; step() is never called.
    """

    def __init__(self, stack):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_FEATURES * stack,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        raise RuntimeError("_SpecEnv exists for its spaces only")


def stack_episode(observations, stack):
    """Reproduce what VecFrameStack hands the policy, step by step.

    Returns (steps, NUM_FEATURES * stack). Row t holds observations
    t-stack+1 .. t, oldest first, with zeros before the episode began --
    which is what VecFrameStack.reset leaves in the buffer.
    """
    steps, features = observations.shape
    out = np.zeros((steps, features * stack), dtype=np.float32)

    for offset in range(stack):
        # Column block `offset` holds the observation from `age` steps ago;
        # the last block (offset == stack - 1) is the current one.
        age = stack - 1 - offset
        if age == 0:
            out[:, offset * features : (offset + 1) * features] = observations
        else:
            out[age:, offset * features : (offset + 1) * features] = observations[:-age]

    return out


def discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def build_dataset(run_id, stack, gamma, racing_only=True):
    episodes = load_episodes(run_id)
    if not episodes:
        raise SystemExit(f"No episodes in run '{run_id}'.")

    all_obs, all_actions, all_returns = [], [], []
    kept_episodes = 0
    dropped_frames = 0

    for episode in episodes:
        steps = load_steps(run_id, episode)
        if steps is None:
            continue

        observations = steps["observations"]
        actions = steps["actions"]
        if len(observations) < stack:
            continue

        stacked = stack_episode(observations, stack)
        returns = discounted_returns(steps["rewards"], gamma)

        if racing_only:
            # The ~70 frames after the machine is destroyed still read as a
            # live race (see env/fzero_env.py). The teacher's inputs there
            # are steering wreckage, and cloning them teaches nothing.
            live = steps["racing"]
            dropped_frames += int((~live).sum())
            stacked, actions, returns = stacked[live], actions[live], returns[live]

        if len(stacked) == 0:
            continue

        all_obs.append(stacked)
        all_actions.append(actions)
        all_returns.append(returns)
        kept_episodes += 1

    if not all_obs:
        raise SystemExit(
            f"Run '{run_id}' has summaries but no per-step arrays to learn from."
        )

    observations = np.concatenate(all_obs)
    actions = np.concatenate(all_actions).astype(np.int64)
    returns = np.concatenate(all_returns)

    print(f"{kept_episodes} episodes, {len(observations)} steps "
          f"({dropped_frames} dropped as not-racing)")
    return observations, actions, returns


def action_mix(actions):
    """What the teacher actually does, so a degenerate clone is visible.

    If one action dominates, a policy that always picks it scores well on
    accuracy while having learned nothing -- worth seeing before believing
    any number below.
    """
    names = ("steer", "shoulder")
    lines = []
    for dim, name in enumerate(names):
        counts = np.bincount(actions[:, dim], minlength=ACTION_DIMS[dim])
        share = counts / counts.sum()
        lines.append(f"  {name:9s} " + "  ".join(f"{v * 100:5.1f}%" for v in share))
    return "\n".join(lines)


def train(model, observations, actions, returns, epochs, batch_size, lr, device):
    policy = model.policy
    policy.train()
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    split = int(len(observations) * (1.0 - VALIDATION_FRACTION))
    order = np.random.permutation(len(observations))
    train_idx, valid_idx = order[:split], order[split:]

    obs_t = torch.as_tensor(observations, device=device)
    act_t = torch.as_tensor(actions, device=device)
    ret_t = torch.as_tensor(returns, device=device)

    print(f"\ntraining on {len(train_idx)} steps, validating on {len(valid_idx)}\n")
    print(f"{'epoch':>5} {'loss':>9} {'val loss':>9} "
          f"{'steer':>7} {'shoulder':>9}")

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        total = 0.0
        batches = 0

        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            obs_b = obs_t[batch]

            distribution = policy.get_distribution(obs_b)
            # Sum over the three action dimensions, which is what
            # MultiCategoricalDistribution.log_prob already does.
            policy_loss = -distribution.log_prob(act_t[batch]).mean()

            values = policy.predict_values(obs_b).squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(values, ret_t[batch])

            # Scaled down: returns here run to the hundreds, and unscaled the
            # value term is orders of magnitude larger than the policy term
            # and the clone never happens.
            loss = policy_loss + 0.01 * value_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total += policy_loss.item()
            batches += 1

        with torch.no_grad():
            valid_obs = obs_t[valid_idx]
            valid_act = act_t[valid_idx]
            distribution = policy.get_distribution(valid_obs)
            val_loss = -distribution.log_prob(valid_act).mean().item()

            # MultiCategoricalDistribution.distribution is one
            # torch.distributions.Categorical per action dimension, so the
            # most likely action comes off its logits.
            predicted = torch.stack(
                [
                    categorical.logits.argmax(dim=-1)
                    for categorical in distribution.distribution
                ],
                dim=1,
            )
            agree = (predicted == valid_act).float().mean(dim=0).cpu().numpy()

        print(f"{epoch + 1:5d} {total / max(batches, 1):9.4f} {val_loss:9.4f} "
              f"{agree[0] * 100:6.1f}% {agree[1] * 100:8.1f}%")

    return agree


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run id from record_teacher.py")
    parser.add_argument("--name", default="bc_ppo_fzero", help="model to write")
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--keep-dead-frames", action="store_true",
                        help="do not drop frames where the race was not live")
    args = parser.parse_args()

    # Same reasoning as training/train.py: this policy is small enough that
    # torch's threading is pure overhead.
    torch.set_num_threads(1)

    observations, actions, returns = build_dataset(
        args.run, args.frame_stack, args.gamma, racing_only=not args.keep_dead_frames
    )

    print("\nteacher action mix:")
    print(action_mix(actions))

    env = DummyVecEnv([lambda: _SpecEnv(args.frame_stack)])
    model = PPO("MlpPolicy", env, verbose=0)

    agree = train(
        model,
        observations,
        actions,
        returns,
        args.epochs,
        args.batch_size,
        args.lr,
        model.device,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, args.name)
    model.save(path)

    print(f"\nSaved {path}.zip")
    print("Agreement with the teacher on held-out steps: "
          f"steer {agree[0] * 100:.1f}%, shoulder {agree[1] * 100:.1f}%")
    print(f"\nContinue with PPO:\n    train_rl.bat 8 500000 --resume --name {args.name}")


if __name__ == "__main__":
    main()
