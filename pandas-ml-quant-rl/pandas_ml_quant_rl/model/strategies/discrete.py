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
        super().__init__(Discrete(3), (1,))
        self.long_columns = long
        self.short_columns = short
        self.return_label = return_label
        self.max_loos = max_loos
        self.total_return = 0

    def trade_reward(self, action, label, sample_weight, gross_loss) -> Tuple[float, bool]:
        trade_return = label[self.return_label]

        if action == 0:
            reward = -0.00001
        elif action == 1:
            reward = trade_return
        else:
            reward = -trade_return

        self.total_return += reward
        return reward, self.total_return < self.max_loos

    def reset(self):
        self.total_return = 0

    def current_state(self) -> np.ndarray:
        # FIXME
        pass

    def current_available_actions(self):
        # FIXME
        pass



