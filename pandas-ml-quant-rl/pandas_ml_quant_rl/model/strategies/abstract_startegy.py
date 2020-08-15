from typing import Tuple

import numpy as np
import pandas as pd
from gym import Space


class Strategy(object):

    def __init__(self, action_space: Space, state_shape: Tuple[int]):
        self.action_space = action_space
        self.state_shape = state_shape

    def trade_reward(self, action, label: pd.Series, sample_weight: pd.Series, gross_loss: pd.Series) -> Tuple[float, bool]:
        """

        :param action:
        :param label:
        :param sample_weight:
        :param gross_loss:
        :return: during training we execute the action and calculate the immediate return of this action alongside
        the information if we can continue trading or not (i.e. we got bust).
        """
        return 0, False

    def current_state(self) -> np.ndarray:
        """
        :return: we return the sate of the current portfolio which means a 2D matrix of i assets and indicator columns like
        percentage long, percentage short, volatility, and eventually more such indicators. Eventually we also return measures
        of the whole portfolio history like return, volatility, sharp ratio etc.: [assets, indicators]

        Eventually a 3D matrix containing a history of n portfolio states [timestepes, assets, indicators]

        """
        return np.zeros(self.state_shape)

    def current_available_actions(self) -> Tuple[int]:
        """

        :return: the set of actions available in the current state i.e. in a long only portfolio with a position only
        sell or hold are actual possible actions from this state
        """
        return self.action_space.shape

    def reset(self):
        """
        When we start a new episode we need to reset the strategy parameters as well
        :return:
        """
        pass