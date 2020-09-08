from typing import Tuple, Dict

import gym
import numpy as np

from pandas_ml_quant_rl.model.environment import Environment


class CartPoleWrappedEnv(Environment):

    def __init__(self, auto_render_after_steps=None):
        self._cart_pole_env = gym.make("CartPole-v0")
        self.action_space = self._cart_pole_env.action_space
        self.observation_space = self._cart_pole_env.observation_space
        self.auto_render_after_steps = auto_render_after_steps
        self._total_reward = 0
        self._step = 0

    def as_train(self) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        return self

    def as_test(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        return self

    def as_predict(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        return self

    def sample_action(self, probs=None):
        return np.random.choice(self._cart_pole_env.action_space.n, p=probs)

    def step(self, action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        state, reward, done, info = self._cart_pole_env.step(action)
        self._total_reward += reward
        self._step += 1

        return (state, np.array([])), reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._total_reward = 0
        self._step = 0

        state = self._cart_pole_env.reset()
        return (state, np.array([]))

    def get_screen(self):
        return self._cart_pole_env.render('rgb_array')