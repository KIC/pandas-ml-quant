from typing import Iterable, Dict, Callable, Tuple, Union

from pandas_ml_common.utils.column_lagging_utils import lag_columns
from pandas_ta_quant._decorators import *


@for_each_top_level_row
def ta_rnn(df: Union[pd.Series, pd.DataFrame],
           feature_lags: Iterable[int],
           lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
           return_min_required_samples=False
           ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:

    # lag columns
    dff = lag_columns(df, feature_lags, lag_smoothing)

    # drop all rows which got nan now
    dff = dff.dropna()

    if return_min_required_samples:
        return dff, len(df) - len(dff)
    else:
        return dff

