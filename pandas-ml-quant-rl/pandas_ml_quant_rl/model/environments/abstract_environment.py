from typing import Tuple, Dict

import gym
import numpy as np
from matplotlib.figure import Figure

from pandas_ml_quant_rl.model.strategies.abstract_startegy import Strategy


class Environment(gym.Env):

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strategy = strategy
        self.action_space = strategy.action_space

    def sample_action(self, probs=None):
        return self.strategy.sample_action(probs)

    def as_train(self) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def as_test(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def as_predict(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def step(self, action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        raise NotImplementedError()

    def render(self, nmode) -> Figure:
        raise NotImplementedError()


class SpaceUtils(object):

    @staticmethod
    def unbounded_box(shape):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)

    @staticmethod
    def unbounded_tuple_boxes(*shapes):
        ts = tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape) for shape in shapes])
        return gym.spaces.Tuple(ts)
