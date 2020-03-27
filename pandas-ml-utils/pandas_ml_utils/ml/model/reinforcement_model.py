import logging
from typing import Callable, Tuple

import gym
import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import DataGenerator

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

            # initialize palace holder variables
            # TODO exclude these variables from serialization
            self.data_generator = None
            self.last_reward = None
            self.current_index = 0
            self.last_index = -1
            self.features = None
            self.labels = None

        @property
        def reward(self):
            return self.last_reward

        def reset(self) -> np.ndarray:
            # get a new data set from the generator
            # NOTE the total numbers which can be drawn from the generator must be greater or equal to the total number
            # of iteration of the learning agent
            self.features, _, self.labels, _, _, _ = next(self.data_generator)

            # reset indices and max timesteps
            self.current_index = 0
            self.last_index = len(self.features)

            # return the very first observation
            return self.next_observation(self.current_index, self.features[self.current_index])

        def step(self, action) -> Tuple[np.ndarray, float, bool]:
            # initialize reward for this action
            done = False
            reward = 0

            # try to execute the action, if it fails with an exception we are done with this session
            try:
                i = self.current_index
                reward = self.take_action(action, i, self.features[i], self.labels[i])
                self.last_reward = reward
            except Exception:
                done = True

            # now update the observation of the environment
            self.current_index += 1
            i_new = self.current_index
            done = done or i_new >= self.last_index
            observation = self.next_observation(i_new, self.features[i_new]) if not done else None

            # return the new observation the reward of this action and whether the session is over
            return observation, reward, done

        def take_action(self, action, idx, features, labels) -> float:
            pass

        def next_observation(self, idx, x) -> np.ndarray:
            pass

        def set_data_generator(self, generator: DataGenerator, **kwargs):
            self.data_generator = generator.sample()
            self.reset()

    def __init__(self,
                 reinforcement_model_provider: Callable[[DataFrameGym], RLModel],
                 gym: DataFrameGym,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.reinforcement_model_provider = reinforcement_model_provider
        self.rl_model = call_callable_dynamic_args(reinforcement_model_provider, gym, **self.kwargs, **kwargs)
        self.gym = gym

    def plot_loss(self):
        # FIXME plot loss .. .somehow ... i.e. use callbacks and collect the rewards ..
        pass

    def fit(self, data: DataGenerator, **kwargs) -> float:
        # initialize the training and test gym
        # use a DummyVecEnv to enable parallel training

        self.gym.set_data_generator(data, **self.kwargs)
        call_callable_dynamic_args(self.rl_model.learn, **self.kwargs, **kwargs)

        return self.gym.reward

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        data = DataGenerator((x, x, None), (x, x, None))
        self.gym.set_data_generator(data, **self.kwargs)
        prediction = []

        for i in range(len(x)):
            obs = self.gym.reset()
            action, state = self.rl_model.predict(obs)
            prediction.append(action)

            obs, reward, done, _ = self.gym.step(action)
            self.gym.render()

        return np.array(prediction)

    def __getstate__(self):
        # FIXME need to be implemented
        pass

    def __setstate__(self, state):
        # FIXME need to be implemented
        pass

    def __call__(self, *args, **kwargs):
        # create a new version of this model
        return ReinforcementModel(
            self.reinforcement_model_provider,
            self.gym,
            self.features_and_labels,
            self.summary_provider,
            **self.kwargs, **kwargs
        )
