from typing import Tuple

import gym
import numpy as np

from pandas_ml_common import pd


class Environment(gym.Env):

    def __init__(self):
        super().__init__()

    def as_train(self) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def as_test(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def as_predict(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SpaceUtils(object):

    @staticmethod
    def unbounded_box(shape):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)

    @staticmethod
    def unbounded_tuple_boxes(*shapes):
        ts = tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape) for shape in shapes])
        return gym.spaces.Tuple(ts)
