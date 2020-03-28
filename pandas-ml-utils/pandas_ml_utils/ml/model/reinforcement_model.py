import logging
from typing import Callable, Tuple, Generator, List, Dict, Any

import gym
import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler
from stable_baselines.common.vec_env import DummyVecEnv

_log = logging.getLogger(__name__)


class ReinforcementModel(Model):

    class RLModel(object):
        def learn(self, total_timesteps, callback=None,
                  log_interval=100, tb_log_name="run", reset_num_timesteps=True):
            pass

        def predict(self, observation, state=None, mask=None, deterministic=False):
            pass

        def get_env(self):
            pass

        def set_env(self, env):
            pass

    class DataFrameGym(gym.Env):

        def __init__(self,
                     action_space: gym.spaces.Space,
                     observation_space: gym.spaces.Space):
            super().__init__()
            self.action_space = action_space
            self.observation_space = observation_space

            # initialize palace holder variables
            self.reward_history = [[]]
            # TODO exclude these variables from serialization
            self.data_generator: Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None] = None
            self.last_reward: float = None
            self.current_index: int = 0
            self.last_index: int = -1
            self.train: Tuple[List[np.ndarray]] = None
            # self.test: Tuple[List[np.ndarray]] = None

        @property
        def reward(self):
            return self.reward_history[:-1] if len(self.reward_history[-1]) <= 0 else self.reward_history

        @property
        def nr_of_sessions(self):
            return len(self.reward_history) - (1 if len(self.reward_history[-1]) <= 0 else 0)

        def reset(self) -> np.ndarray:
            # get a new data set from the generator
            # NOTE the total numbers which can be drawn from the generator must be greater or equal to the total number
            # of iteration of the learning agent
            self.train, _ = next(self.data_generator)

            # reset indices and max timesteps and add a new reward history
            self.current_index = 0
            self.last_index = len(self.train[0])
            if len(self.reward_history[-1]) > 0:
                self.reward_history.append([])

            # return the very first observation
            i = self.current_index
            return self.next_observation(i, *[t[i] if t is not None else None for t in self.train])

        def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
            # initialize reward for this action
            done = False
            reward = 0

            # try to execute the action, if it fails with an exception we are done with this session
            try:
                if action.shape != self.action_space.shape:
                    action = action[-1]

                i = self.current_index
                reward = self.take_action(action, i, *[t[i] if t is not None else None for t in self.train])
                self.reward_history[-1].append(reward)
                self.last_reward = reward
            except StopIteration:
                done = True

            # now update the observation of the environment
            self.current_index += 1
            i_new = self.current_index
            done = done or i_new >= self.last_index
            observation = self.next_observation(i_new, *[t[i_new] if t is not None else None for t in self.train]) if not done else None

            # return the new observation the reward of this action and whether the session is over
            return observation, reward, done, {}

        def take_action(self,
                        action,
                        idx: int,
                        features: np.ndarray,
                        labels: np.ndarray,
                        targets: np.ndarray,
                        weights: np.ndarray) -> float:
            pass

        def next_observation(self,
                             idx: int,
                             features: np.ndarray,
                             labels: np.ndarray,
                             targets: np.ndarray,
                             weights: np.ndarray) -> np.ndarray:
            pass

        def set_data_generator(self, sampler: Sampler, **kwargs):
            self.data_generator = sampler.sample()
            return self.reset()

    def __init__(self,
                 reinforcement_model_provider: Callable[[Any], RLModel],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.reinforcement_model_provider = reinforcement_model_provider
        self.rl_model = call_callable_dynamic_args(reinforcement_model_provider, **self.kwargs, **kwargs)
        self.reward_history = None
        self.env = self.rl_model.get_env()

    def fit(self, sampler: Sampler, **kwargs) -> float:
        # provide the data generator for every environment
        envs = self.rl_model.get_env().envs

        for env in envs:
            env.set_data_generator(sampler, **self.kwargs)

        call_callable_dynamic_args(self.rl_model.learn, **self.kwargs, **kwargs)

        # collect statistics of all environments
        envs = self.rl_model.get_env().envs
        self.reward_history = [np.array(hist) for env in envs for hist in env.reward_history]
        average_reward = np.array([hist.mean() for hist in self.reward_history]).mean()
        sessions = np.array([env.nr_of_sessions for env in envs]).sum()
        _log.info(f"average reward = {average_reward} in {sessions}")

        return average_reward

    def predict(self, sampler: Sampler, **kwargs) -> np.ndarray:
        # set a fresh env
        self.rl_model.set_env(self.env)
        samples, _ = sampler.nr_of_source_events
        envs = self.rl_model.get_env().envs[:1]
        obs = [env.set_data_generator(sampler, **self.kwargs) for env in envs]
        prediction = []

        for i in range(samples):
            action, state = self.rl_model.predict(obs)
            prediction.append(action)

            obs, reward, done, _ = zip(*[env.step(action) for env in envs])
            for env, dne in zip(envs, done):
                call_callable_dynamic_args(env.render, kwargs)

                # eventually the agent got killed and needs to reset
                if dne and i < samples - 1:
                    env.reset()

        return np.array(prediction)

    def plot_loss(self):
        import matplotlib.pyplot as plt

        for i, session in enumerate(self.reward_history):
            s = pd.Series(session).cumsum()
            plt.plot(s, label=f'reward {i}')

        # only plot legend if it fits on the plot
        if len(self.reward_history) <= 5:
            plt.legend(loc='best')

    def __getstate__(self):
        # FIXME need to be implemented
        # rl_model.get_env() -> save the first env!
        pass

    def __setstate__(self, state):
        # FIXME need to be implemented
        # rl_model.set_env(saved <- envs)
        pass

    def __call__(self, *args, **kwargs):
        # create a new version of this model
        return ReinforcementModel(
            self.reinforcement_model_provider,
            self.features_and_labels,
            self.summary_provider,
            **self.kwargs, **kwargs
        )
