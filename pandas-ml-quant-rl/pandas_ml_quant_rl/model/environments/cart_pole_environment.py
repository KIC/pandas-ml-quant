from typing import Tuple, Dict

import gym
import numpy as np
from matplotlib.figure import Figure

from .abstract_environment import Environment


class CartPoleWrappedEnv(Environment):

    def __init__(self, auto_render_after_steps=None):
        self._cart_pole_env = gym.make("CartPole-v0")
        self.action_space = self._cart_pole_env.action_space
        self.observation_space = self._cart_pole_env.observation_space
        self.auto_render_after_steps = auto_render_after_steps
        self._total_reward = 0
        self._step = 0
        self._fig = None

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

        # eventually render each n steps
        if self.auto_render_after_steps is not None and (self._step % self.auto_render_after_steps == 0 or done):
            self.render()

        self._step += 1
        return (state, np.array([])), reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._total_reward = 0
        self._step = 0
        self._fig = None

        state = self._cart_pole_env.reset()
        return (state, np.array([]))

    def render(self, mode='human') -> Figure:
        import matplotlib
        import matplotlib.pyplot as plt

        try:
            if 'inline' in matplotlib.get_backend():
                from IPython.display import clear_output
                clear_output(wait=True)
            if self._fig is None:
                self._fig = plt.figure()

            screen = self._cart_pole_env.render('rgb_array')
            plt.clf()
            plt.title(f"last reward: {self._total_reward}")
            plt.imshow(screen)
            plt.show()
        except:
            # allow interrupting by the user
            pass

        return self._fig


