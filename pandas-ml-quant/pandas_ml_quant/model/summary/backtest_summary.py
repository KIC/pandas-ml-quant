from pandas_ml_quant.trading.transaction_log import StreamingTransactionLog
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME
from pandas_ml_utils.ml.summary import Summary
from pandas_ml_common import Typing


class BacktestSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, **kwargs):
        super().__init__(df, **kwargs)

    def calculate_performance(self, initial_capital=100, prediction_name="0", target_name="Close"):
        log = StreamingTransactionLog()
        self.df[PREDICTION_COLUMN_NAME, prediction_name].apply(lambda action: log.rebalance(action * initial_capital))
        perf = log.evaluate(self.df[TARGET_COLUMN_NAME, target_name].rename("prices"))

        return perf

