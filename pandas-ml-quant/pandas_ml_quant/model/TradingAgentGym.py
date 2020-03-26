from pandas_ml_utils import ReinforcementModel
from gym.spaces import MultiDiscrete, Box
import numpy as np


class TradingAgent(ReinforcementModel.DataFrameGym):

    def __init__(self,
                 input_shape,
                 initial_capital=100000,
                 commission=lambda size: 0.025):
        super().__init__(MultiDiscrete([3, 10]), # [buy|hold|sell, n/10]
                         Box(low=-1, high=1, shape=input_shape)) # FIXME what shape? we also need historic trades?

        self.initial_capital = initial_capital
        self.commission_calculator = commission
        self.capital = initial_capital
        self.trades = []

        # TODO add some observations to the observation space i.e.
        #   our net worth, the amount of BTC bought or sold, and the total amount in USD we’ve spent on or received
        #   from those BTC.

    def reset(self):
        self.trades = []
        self.capital = self.initial_capital
        return super().reset()

    def take_action(self, action, idx, x, y) -> float:
        # FIXME currently returns fake award
        # TODO throw a value error if bankrupt
        return 0.1

    def next_observation(self, idx, x) -> np.ndarray:
        # FIXME currently returns only the features
        return x

    def render(self, mode='human'):
        if mode == 'system':
            # TODO print something
            pass
        elif mode == 'notebook':
            # TODO plot something using matplotlib
            pass
        elif mode == 'human':
            # TODO plot something using matplotlib
            pass


class TradingAgentModel(ReinforcementModel):

    def __init__(self):
        # TradingReward(initial_capital=100000)
        pass
