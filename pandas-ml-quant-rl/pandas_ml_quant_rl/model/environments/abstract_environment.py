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
