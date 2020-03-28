from typing import Callable

from pandas_ml_utils import ReinforcementModel, FeaturesAndLabels
from gym.spaces import MultiDiscrete, Box
from pandas_ml_common import np, pd, Typing

from pandas_ml_utils.ml.summary import Summary


class TradingAgentGym(ReinforcementModel.DataFrameGym):

    def __init__(self,
                 input_shape,
                 initial_capital=100000,
                 commission=lambda size: 0.025):
        super().__init__(MultiDiscrete([3, 10]), # [buy|hold|sell, n/10]
                         Box(low=-1, high=1, shape=input_shape)) # FIXME what shape? we also need historic trades?

        self.initial_capital = initial_capital
        self.commission_calculator = commission

        # place holder variables setted by reset function
        # TODO add some observations to the observation space i.e.
        #   our net worth, the amount of BTC bought or sold, and the total amount in USD we’ve spent on or received
        #   from those BTC.
        self.cash = None
        self.position = 0
        self.trades = None
        self.net = 0

    def reset(self) -> np.ndarray:
        self.trades = []
        self.cash = self.initial_capital
        return super().reset()

    def take_action(self,
                    action,
                    idx: int,
                    features: np.ndarray,
                    labels: np.ndarray,
                    targets: np.ndarray,
                    weights: np.ndarray) -> float:
        action_type = action[0]
        amount = action[1] / 10
        net = self.cash + self.position * targets[0]

        if action_type < 1:
            # hold
            pass
        elif action_type < 2:
            # buy
            if self.position < 0.999:
                self.position += amount
                self.cash -= targets[0] * amount
        elif action_type < 3:
            # sell
            if self.position > 0:
                self.position -= amount
                self.cash += targets[0] * amount
        else:
            raise ValueError(f"unknown action type {action_type}!")

        new_net = self.cash + self.position * targets[0]
        self.net = new_net

        if net < self.initial_capital * 0.8:
            raise StopIteration("lost more then 20%")

        return new_net / net - 1

    def next_observation(self,
                         idx: int,
                         features: np.ndarray,
                         labels: np.ndarray,
                         targets: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
        # FIXME currently returns only the features, but we also want to return some net worth, history, ...
        return features

    def render(self, mode='human'):
        if mode == 'system':
            print(self.net)
        elif mode == 'notebook':
            # TODO plot something using matplotlib
            pass
        elif mode == 'human':
            # TODO plot something using matplotlib
            pass

