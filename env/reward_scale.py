"""Constant reward rescaling, for value-based learners.

WHY A CONSTANT AND NOT VecNormalize. train.py normalises PPO's reward with
VecNormalize, which tracks a running standard deviation. That is fine on-policy,
where every batch is fresh. It is wrong for DQN: rewards are normalised before
they enter the replay buffer, so as the running statistic drifts the buffer ends
up holding rewards from several different scales at once, and the Q-targets it
regresses toward are mutually inconsistent. A fixed divisor is stationary and
therefore replay-safe.

WHY IT MATTERS AT ALL. Q-learning regresses onto r + gamma * max Q(s', a'), so
the network output has to span the range of returns. Measured over 26675
episodes of potential-mode DQN, episode return ran -284 to +336 (1st to 99th
percentile) with a standard deviation of 96. The difference between actions at
a given state is a couple of points, so picking an argmax meant resolving well
under 1% relative differences on a large output. Dividing by 50 puts the same
returns at a standard deviation of 1.9 and a 1-99 range of -5.7 to 6.7.

This wrapper deliberately sits ABOVE RecordingWrapper, so the run store and the
dashboards keep native units and stay comparable with every earlier run. Only
the learner sees the scaled value.
"""

import gymnasium as gym

# 1/50. See the module docstring for the measurement behind it.
DEFAULT_REWARD_SCALE = 0.02


class RewardScale(gym.Wrapper):
    def __init__(self, env, scale=DEFAULT_REWARD_SCALE):
        super().__init__(env)
        scale = float(scale)
        if scale <= 0:
            raise ValueError(f"reward scale must be positive, got {scale}")
        self.scale = scale

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward * self.scale, terminated, truncated, info
