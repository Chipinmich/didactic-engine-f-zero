"""Present the no-brake steer/lean space as Discrete(9), for DQN.

FZeroEnv always holds acceleration. Its action is MultiDiscrete([3, 3]):
steer (straight, left, right) and lean (none, left, right). DQN needs one
Discrete space, so this wrapper maps 0..8 to the corresponding pair.
"""

import gymnasium as gym
from gymnasium import spaces


class FlatAction(gym.Wrapper):
    """MultiDiscrete([3, 3]) presented as Discrete(9)."""

    ACTIONS = 9

    LEANS = 3

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(self.ACTIONS)

    @classmethod
    def flatten(cls, steer, lean):
        """(steer, lean) -> the Discrete code, the inverse of unflatten.

        Exposed because the safety assist can change the action after the
        policy has chosen it, and the replay buffer has to record the pair
        that actually reached the game rather than the one that was asked
        for. See TruthfulActionDQN in training/train_dqn.py.
        """
        steer, lean = int(steer), int(lean)
        if not 0 <= steer < cls.LEANS or not 0 <= lean < cls.LEANS:
            raise ValueError(f"steer and lean must be 0..2, got {(steer, lean)}")
        return steer * cls.LEANS + lean

    @classmethod
    def unflatten(cls, code):
        code = int(code)
        if not 0 <= code < cls.ACTIONS:
            raise ValueError(f"action must be 0..{cls.ACTIONS - 1}, got {code}")
        return code // cls.LEANS, code % cls.LEANS

    def step(self, action):
        return self.env.step(self.unflatten(action))
