import logging
import gym
from copy import deepcopy
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
_log = logging.getLogger(__name__)


class ReinforcementModel(Model):

    class RLModel(object):
        def learn(self, total_timesteps, callback=None,
                  log_interval=100, tb_log_name="run", reset_num_timesteps=True):
            pass

        def predict(self, observation, state=None, mask=None, deterministic=False):
            pass

    class DataFrameGym(gym.Env):

        def __init__(self,
                     action_space: gym.spaces.Space,
                     observation_space: gym.spaces.Space):
            super().__init__()
            self.action_space = action_space
            self.observation_space = observation_space
            self.current_index = 0

            self.x = None
            self.y = None
            self.last_index = -1

        def reset(self):
            self.current_index = 0
            return self.next_observation(self.current_index, self.x[self.current_index])

        def step(self, action):
            done = False
            reward = 0

            try:
                i = self.current_index
                reward = self.take_action(action, i, self.x[i], self.y[i])
            except ValueError:
                done = True

            self.current_index += 1
            i_next = self.current_index
            done = done or i_next >= self.last_index
            observation = self.next_observation(i, self.x[i_next]) if not done else None

            return observation, reward, done

        def take_action(self, action, idx, x, y) -> float:
            pass

        def next_observation(self, idx, x) -> np.ndarray:
            pass

        def __call__(self, *args, **kwargs):
            self.x, self.y = args
            self.last_index = len(self.x)
            self.reset()
            return self

    def __init__(self,
                 reinforcement_model: Callable[[DataFrameGym], RLModel],
                 gym: DataFrameGym,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.reinforcement_model = reinforcement_model
        self.gym = gym

    def plot_loss(self):
        pass

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray,
            sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
            **kwargs) -> float:
        # initialize the training and test gym
        train_gym = self.gym(x, y, **self.kwargs)
        test_gym = self.gym(x_val, y_val, **self.kwargs)

        model = call_callable_dynamic_args(self.reinforcement_model, train_gym, **self.kwargs, **kwargs)
        call_callable_dynamic_args(model.learn, **self.kwargs, **kwargs)
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass



