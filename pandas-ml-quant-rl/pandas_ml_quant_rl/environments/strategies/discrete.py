from typing import Tuple

import numpy as np
from gym.spaces import Discrete

from pandas_ml_quant_rl.model.environment import Strategy


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
                 max_loos: float = -0.2
                 ):
        super().__init__(Discrete(3), buffer=1, assets=1, indicators=1)
        self.long_columns = long
        self.short_columns = short
        self.max_loos = max_loos
        self.total_return = 0

    def sample_action(self, probs=None):
        action = np.random.choice(self.action_space.n, p=probs) if probs is not None else self.action_space.sample()
        return action

    def trade_reward(self, previous_bar, action, bar):
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


class LongOnly(Strategy):

    def __init__(self,
                 buy: str = 'Close',
                 sell: str = 'Close',
                 max_loos: float = -0.2,
                 max_holding_period: int = None
                 ):
        # we have 4 actions hold, buy, sell, swing
        super().__init__(Discrete(4), buffer=1, assets=1, indicators=1)
        if max_holding_period is not None: raise NotImplemented()  # FIXME implement max holding period and eventuall also implement something like stop loss orders
        self.buy_col = buy
        self.sell_col = sell
        self.max_loos = max_loos
        self.max_holding_period = max_holding_period

        self.positioning = False
        self.total_return = 0
        self.opening_value = None

    def sample_action(self, probs=None):
        if probs is None: probs = np.ones(self.action_space.n)
        if not isinstance(probs, np.ndarray): probs = np.array(probs)

        if self.positioning:
            # we are long so buy is not allowed
            probs[1] = 0

        if not self.positioning:
            # we are not positioned so sell is not allowed
            probs[2] = 0

        # rescale probabilities
        probs = probs / probs.sum()

        action = np.random.choice(self.action_space.n, p=probs)
        return action

    def trade_reward(self, previous_bar, action, bar) -> Tuple[np.ndarray, float, bool]:
        # FIXME ....
        reward = 0
        row_to_buffer = np.array([[reward]])
        return self._roll_state_buffer(row_to_buffer), reward, self.total_return < self.max_loos

    def reset(self):
        super().reset()
        self.positioning = False
        self.total_return = 0
        self.opening_value = None


class LongShortSwing(Strategy):

    def __init__(self,
                 buy: str = 'Close',
                 sell: str = 'Close',
                 max_loos: float = -0.2,
                 max_holding_period: int = None
                 ):
        # we have 4 actions hold, buy, sell, swing
        super().__init__(Discrete(4), buffer=1, assets=1, indicators=1)
        if max_holding_period is not None: raise NotImplemented()  # FIXME implement max holding period and eventuall also implement something like stop loss orders
        self.buy_col = buy
        self.sell_col = sell
        self.max_loos = max_loos
        self.max_holding_period = max_holding_period

        self.positioning = 0
        self.total_return = 0
        self.opening_value = None

    def sample_action(self, probs=None):
        if probs is None: probs = np.ones(self.action_space.n)
        if not isinstance(probs, np.ndarray): probs = np.array(probs)

        if self.positioning > 0:
            # we are long so buy is not allowed
            probs[1] = 0
        elif self.positioning < 0:
            # we are short so sell is not allowed
            probs[2] = 0
        elif self.positioning == 0:
            # we cant swing if we don't have an open position yet
            probs[3] = 0

        # rescale probabilities
        probs = probs / probs.sum()

        action = np.random.choice(self.action_space.n, p=probs)
        return action

    def trade_reward(self, previous_bar, action, bar):
        if action == 1 and self.positioning != 1:
            # enter a new long position, reward is 0
            self.opening_value = bar[self.buy_col]
            if self.positioning < 0:
                self.positioning = 0
                reward = np.log(self.opening_value / bar[self.buy_col])
            else:
                self.positioning = 1
                reward = 0
        elif action == 2 and self.positioning != -1:
            # enter a new long position, reward is 0
            self.opening_value = bar[self.sell_col]
            if self.positioning > 0:
                self.positioning = 0
                reward = np.log(bar[self.sell_col] / self.opening_value)
            else:
                self.positioning = -1
                reward = 0
        elif action == 3 and self.positioning != 0:
            # reward is revenue of closing the position
            if self.positioning > 0:
                pct = (bar[self.sell_col] / self.opening_value)
                self.opening_value = bar[self.sell_col]
            else:
                pct = (self.opening_value / bar[self.buy_col])
                self.opening_value = bar[self.buy_col]

            self.positioning = -self.positioning
            reward = np.log(pct)
        else:
            # reward is the position value change or -0.00001 in case of no open position
            # if we exceed the maximum holding period we should also use the closing posiiton reward
            if self.positioning == 0:
                reward = -0.0001
            else:
                pct = (bar[self.sell_col] / self.opening_value) if self.positioning > 0 else (self.opening_value / bar[self.buy_col])
                reward = np.log(pct)

        self.total_return += reward

        # remember our current positioning -> we should use this information in the network such that the agent
        # learns which actions are executable given a long/sort position
        return self._roll_state_buffer(np.array([[self.positioning]])), reward, self.total_return < self.max_loos

    def reset(self):
        super().reset()
        self.positioning = 0
        self.total_return = 0
        self.opening_value = None


# TODO eventually support stop orders for risk mamagement
#  reward could also be sharp ratio or sortino ratio
#  use the log return change of the portfolio net worth, eventually only use excess log returns (demean the return)
