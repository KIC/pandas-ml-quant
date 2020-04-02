from typing import Tuple, Callable

import pandas as pd

from pandas_ml_quant.trading.transaction_log import StreamingTransactionLog
from pandas_ml_common import Typing
from pandas_ml_common.utils import has_indexed_columns


def ta_backtest(signal: Typing.PatchedDataFrame,
                prices: Typing.PatchedPandas,
                action: Callable[[pd.Series], Tuple[int, float]],
                slippage: Callable[[float], float] = lambda _: 0):
    if has_indexed_columns(signal):
        assert len(signal.columns) == len(prices.columns), "Signal and Prices need the same shape!"
        res = pd.DataFrame({}, index=signal.index, columns=pd.MultiIndex.from_product([[], []]))

        for i in range(len(signal.columns)):
            df = ta_backtest(signal[signal.columns[i]], prices[prices.columns[i]], action, slippage)

            top_level_name = ",".join(prices.columns[i]) if isinstance(prices.columns[i], tuple) else prices.columns[i]
            df.columns = pd.MultiIndex.from_product([[top_level_name], df.columns.to_list()])
            res = res.join(df)

        return res

    assert isinstance(prices, pd.Series), "prices need to be a series!"
    trades = StreamingTransactionLog()

    def trade_log_action(row):
        direction_amount = action(row)
        if isinstance(direction_amount, tuple):
            trades.perform_action(*direction_amount)
        else:
            trades.rebalance(float(direction_amount))

    signal.to_frame().apply(trade_log_action, axis=1, raw=True)
    return trades.evaluate(prices.rename("price"), slippage)