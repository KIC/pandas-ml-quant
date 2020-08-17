from typing import Tuple, Union

import numpy as np

from .abstract_startegy import Strategy
from gym.spaces import Discrete


class BuyOpenSellCloseSellOpenBuyClose(Strategy):
    """
    This is kind of a dummy strategy set where we have 3 actions:
    0: do nothing
    1: buy at the open and sell the same day close
    2: sell the close and buy the same day open
    """

    def __init__(self,
                 long: Tuple[str, str] = ('Open', 'Close'),
                 short: Tuple[str, str] = ('Close', 'Open'),
                 return_label: Union[str, tuple, int] = 0,
                 max_loos: float = -0.2
                 ):
        super().__init__(Discrete(3), buffer=1, assets=1, indicators=1)
        self.long_columns = long
        self.short_columns = short
        self.return_label = return_label
        self.max_loos = max_loos
        self.total_return = 0

    def sample_action(self, probs=None):
        action = np.random.choice(self.action_space.n, p=probs) if probs is not None else self.action_space.sample()
        return action

    def trade_reward(self, action, bar):
        if action == 0:
            reward = -0.00001
        elif action == 1:
            reward = np.log(bar[self.long_columns[1]] / bar[self.long_columns[0]])
        else:
            reward = np.log(bar[self.long_columns[0]] / bar[self.long_columns[1]])

        self.total_return += reward

        # remember one indicator for one asset
        row_to_buffer = np.array([[reward]])
        return self._roll_state_buffer(row_to_buffer), reward, self.total_return < self.max_loos

    def reset(self):
        super().reset()
        self.total_return = 0


