from typing import Dict
from typing import Tuple

import gym
import numpy as np
import pandas as pd
from gym import Space
from matplotlib.figure import Figure

from pandas_ml_quant_rl.renderer.abstract_renderer import Renderer


class Strategy(object):

    def __init__(self,
                 action_space: Space,
                 buffer: int = 1,
                 assets: int = 1,
                 indicators: int = 1,
                 renderer: Renderer = Renderer()
                 ):
        """

        :param action_space: a gmy space defining the possible actions
        :param buffer: size of a buffer which is holding information like executed actions and rewards but
            also information like currently open positions for example. Each strategy needs to decide which information
            is needed. default buffer size = 1
        :param assets: number of assets simultaneously trade-able, default 1
        :param indicators: the number of indicators the buffer can hold like action and rewards would be 2.
            default = 1 (reward)
        """
        self.action_space = action_space
        self.buffer = buffer
        self.assets = assets
        self.indicators = indicators
        self.buffered_state = np.zeros((buffer, assets, indicators))

    def sample_action(self, *args, **kwargs):
        raise NotImplementedError()

    def current_state(self) -> np.ndarray:
        return self.buffered_state

    def trade_reward(self, previous_bar: pd.Series, action, bar: pd.Series) -> Tuple[np.ndarray, float, bool]:
        """

        :param action:
        :param label:
        :param sample_weight:
        :param gross_loss:
        :return: we return a tuple of (state, reward, bust)
        the sate of the portfolio after the action which is a 3D matrix of [timestepes, assets, indicators].
        Each strategy has to implement a set of indicators like rewards, positions weights or sharp ratio etc.

        during training we execute the action and calculate the immediate return of this action alongside
        the information if we can continue trading or not (i.e. we got bust).
        """

        return self.current_state(), 0, False

    def _roll_state_buffer(self, state: np.ndarray) -> np.ndarray:
        """
        A helper function to roll over the state array

        :return: rolled state
        """

        # rollover the current state and store the new data
        self.buffered_state = np.roll(self.buffered_state, -1, 0)
        self.buffered_state[-1] = state
        return self.buffered_state

    def current_available_actions(self) -> Tuple[int]:
        """

        :return: the set of actions available in the current state i.e. in a long only portfolio with a position only
        sell or hold are actual possible actions from this state
        """
        return self.action_space.shape

    def reset(self) -> np.ndarray:
        """
        When we start a new episode we need to reset the strategy parameters as well
        :return:
        """
        self.buffered_state = np.zeros(self.buffered_state.shape)
        return self.current_state()

    def render(self, mode='human'):
        return None


class Environment(gym.Env):

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strategy = strategy
        self.action_space = strategy.action_space

    def sample_action(self, probs=None):
        return self.strategy.sample_action(probs)

    def execute_action(self, action, render_on_axis=None) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        if render_on_axis is not None: self.render(render_on_axis)
        return self.step(action)

    def render(self, mode) -> Figure:
        import matplotlib

        if mode == 'matplotlib':
            import matplotlib.pyplot as plt
            return plt.imshow(self.get_screen())
        elif isinstance(mode, matplotlib.axes.Axes):
            ax = mode
            ax.clear()
            ax.imshow(self.get_screen())
            return ax

    def step(self, action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        raise NotImplementedError()

    def as_train(self) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def as_test(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def as_predict(self, renderer=None) -> Tuple['Environment', Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_screen(self):
        raise NotImplementedError()

