import logging
from typing import Callable, Tuple, Generator, List, Dict, Any

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler

_log = logging.getLogger(__name__)


class ReinforcementModel(Model):
    import gym

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
        import gym

        def __init__(self,
                     action_space: gym.spaces.Space,
                     observation_space: gym.spaces.Space):
            super().__init__()
            self.action_space = action_space
            self.observation_space = observation_space

            # initialize palace holder variables
            # TODO exclude these variables from serialization
            self.reward_history_collector = None
            self.data_generator: Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None] = None
            self.last_reward: float = None
            self.current_index: int = 0
            self.last_index: int = -1
            self.train: Tuple[List[np.ndarray]] = None
            # self.test: Tuple[List[np.ndarray]] = None

        def reset(self) -> np.ndarray:
            # get a new data set from the generator
            # NOTE the total numbers which can be drawn from the generator must be greater or equal to the total number
            # of iteration of the learning agent
            self.train, _ = next(self.data_generator)

            # reset indices and max timesteps and add a new reward history
            self.current_index = 0
            self.last_index = len(self.train[0])
            self.reward_history_collector(None)

            # return the very first observation
            i = self.current_index
            return self.next_observation(i, *[t[i] if t is not None else None for t in self.train])

        def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
            # initialize reward for this action
            done = False
            reward = 0

            # try to execute the action, if it fails with an exception we are done with this session
            try:
                i = self.current_index
                state = [t[i] if t is not None else None for t in self.train]
                interpreted_action = self.interpret_action(action, i, *state)
                reward = self.take_action(interpreted_action, i, *state)
                self.reward_history_collector(reward)
                self.last_reward = reward
            except StopIteration:
                done = True

            # now update the observation of the environment
            self.current_index += 1
            i_new = self.current_index
            done = done or i_new >= self.last_index
            observation = self.next_observation(i_new, *[t[i_new] if t is not None else None for t in self.train]) if not done else None

            # return the new observation the reward of this action and whether the session is over
            return observation, reward, done, {"interpreted_action": interpreted_action}

        def interpret_action(self,
                        action,
                        idx: int,
                        features: np.ndarray,
                        labels: np.ndarray,
                        targets: np.ndarray,
                        weights: np.ndarray = None,
                        gross_loss: np.ndarray = None) -> float:
            return action

        def take_action(self,
                        action,
                        idx: int,
                        features: np.ndarray,
                        labels: np.ndarray,
                        targets: np.ndarray,
                        weights: np.ndarray = None,
                        gross_loss: np.ndarray = None) -> float:
            pass

        def next_observation(self,
                             idx: int,
                             features: np.ndarray,
                             labels: np.ndarray,
                             targets: np.ndarray,
                             weights: np.ndarray = None,
                             gross_loss: np.ndarray = None) -> np.ndarray:
            pass

        def set_data_generator(self, sampler: Sampler, reward_history_collector, **kwargs):
            self.data_generator = sampler.sample()
            self.reward_history_collector = reward_history_collector
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
        self.reward_history = []

        for env in envs:
            rh = []
            self.reward_history.append(rh)
            env.set_data_generator(sampler,
                                   lambda reward: rh[-1].append(reward) if reward is not None else rh.append([]),
                                   **self.kwargs)

        call_callable_dynamic_args(self.rl_model.learn, **self.kwargs, **kwargs)

        # collect statistics of all environments
        latest_rewards = np.array([rh[-1] for erh in self.reward_history for rh in erh if len(rh) > 1])
        return latest_rewards.mean()

    def predict(self, sampler: Sampler, **kwargs) -> np.ndarray:
        # set a fresh env
        self.rl_model.set_env(self.env)
        samples, _ = sampler.nr_of_source_events
        envs = self.rl_model.get_env().envs[:1]
        rh = []

        obs = [env.set_data_generator(sampler,
                                      lambda reward: rh[-1].append(reward) if reward is not None else rh.append([]),
                                      **self.kwargs) for env in envs]

        prediction = []
        done = False

        for i in range(samples):
            if not done:
                action, state = self.rl_model.predict(obs)
                obs, reward, done, info = zip(*[env.step(action) for env in envs])
                prediction.append(info[0]["interpreted_action"])

                for env, dne in zip(envs, done):
                    call_callable_dynamic_args(env.render, kwargs)

                done = any(done)
            else:
                prediction.append(np.nan)

        return np.array(prediction)

    def plot_loss(self, accumulate_reward=False):
        import matplotlib.pyplot as plt
        plots = 0
        mins = []

        for env_hist in self.reward_history:
            for i, session in enumerate(env_hist):
                if len(session) > 1:
                    s = pd.Series(session)
                    if accumulate_reward:
                        s = s.cumsum()

                    mins.append(s.min())
                    plt.plot(s, label=f'reward {i}')
                    plots += 1

        # only plot legend if it fits on the plot
        if plots <= 10:
            plt.legend(loc='best')

        print(f'worst reward: {np.array(mins).min()}')

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
