import logging
from typing import Tuple

from gym.spaces import MultiDiscrete, Box, Discrete

from pandas_ml_common import np, pd
from pandas_ml_quant.trading.transaction_log import StreamingTransactionLog
from pandas_ml_utils import ReinforcementModel

_log = logging.getLogger(__name__)


class TradingAgentGym(ReinforcementModel.DataFrameGym):

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 trading_fraction: int = 10,
                 trading_assets: int = 1,  # later we want the bot to trade one of multiple possible assets
                 allow_short: bool = False,
                 stop_if_lost: float = None,
                 initial_capital: float = 100000,
                 commission=lambda size: 0.025):
        super().__init__(MultiDiscrete([trading_assets, trading_fraction]) if trading_assets > 1 else
                         Discrete(trading_fraction + 1),
                         Box(low=-1, high=1, shape=input_shape)) # FIXME what shape? we also need historic trades?

        self.trading_fraction = trading_fraction
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_if_lost = stop_if_lost
        self.allow_short = allow_short

        if allow_short and (trading_fraction % 2) != 0:
            _log.warning('short trades expect even nr of trading fraction')

        # eventually do not serialize ..
        self.trade_log = StreamingTransactionLog()
        self.current_net = 0

    def reset(self) -> np.ndarray:
        self.trade_log = StreamingTransactionLog()
        self.current_net = 0
        return super().reset()

    def interpret_action(self,
                         action,
                         idx: int,
                         features: np.ndarray,
                         labels: np.ndarray,
                         targets: np.ndarray,
                         weights: np.ndarray = None,
                         gross_loss: np.ndarray = None) -> float:
        if self.allow_short:
            # 0 - 10  ->  -10 - 10
            action -= int(self.trading_fraction / 2) * 2

        # we convert the action in to a target balance we pass to the transaction log
        balance = action / self.trading_fraction / float(targets) * self.initial_capital
        return balance

    def take_action(self,
                    action,
                    idx: int,
                    features: np.ndarray,
                    labels: np.ndarray,
                    targets: np.ndarray,
                    weights: np.ndarray = None,
                    gross_loss: np.ndarray = None) -> float:

        # skip the very first bar
        if idx <= 0:
            self.trade_log.rebalance(0)
            return 0

        # we use a n/fractions as target balance -> 10 shares 20,... shares, ...
        balance = action / self.trading_fraction * self.initial_capital
        self.trade_log.rebalance(balance)

        # new we need to evaluate the portfolio performance
        # where the targets are the prices
        prices = pd.Series(self.train[2][:idx+1].ravel(), name="prices")
        perf = self.trade_log.evaluate(prices, self.commission)
        self.current_net = perf["net"].iloc[-1]

        if self.stop_if_lost is not None and self.current_net < self.initial_capital * (1 - self.stop_if_lost):
            raise StopIteration(f"lost more then {self.stop_if_lost}%")

        return self.calculate_trade_reward(perf)

    def calculate_trade_reward(self, portfolio_performance_log) -> float:
        return portfolio_performance_log["net"].iloc[-1]

    def next_observation(self,
                         idx: int,
                         features: np.ndarray,
                         labels: np.ndarray,
                         targets: np.ndarray,
                         weights: np.ndarray = None,
                         gross_loss: np.ndarray = None) -> np.ndarray:
        # currently returns only the features, but we also want to return some net worth, history, ...
        # for CNNs input must be [0, 255] and channel last, this means we need to reshape our feature space
        # return features.swapaxes(0, 1).swapaxes(1, 2) * 255
        pass

    def render(self, mode='human'):
        if mode == 'system':
            print(self.current_net)
        elif mode == 'notebook':
            # TODO plot something using matplotlib
            pass
        elif mode == 'human':
            # TODO plot something using matplotlib
            pass

