from typing import Tuple, Callable

import pandas as pd

from pandas_ml_quant.trading.transaction_log import TransactionLog
from pandas_ml_common import Typing


def ta_backtest(signal: Typing.PatchedDataFrame, prices: pd.Series, action: Callable[[pd.Series], Tuple[int, float]]):
    assert isinstance(prices, pd.Series), "prices need to be a series!"

    trades = TransactionLog()
    signal.to_frame().apply(lambda row: trades.action(*action(row)), axis=1, raw=True)
    return trades.evaluate(prices)