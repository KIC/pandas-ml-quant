from typing import Union, Iterable, Dict, Callable, Tuple
import pandas as pd
from sortedcontainers import SortedDict


def lag_columns(df: Union[pd.Series, pd.DataFrame],
                feature_lags: Iterable[int],
                lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                multi_index: bool = True
               ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:

    df = df.to_frame()
    dff = pd.DataFrame({}, index=df.index)

    # return RNN shaped 3D arrays
    for feature in df.columns:
        feature_series = df[feature]
        smoothers = None

        # smooth out feature if requested
        if lag_smoothing is not None:
            smoothers = SortedDict({lag: smoother(feature_series.to_frame())
                                    for lag, smoother in lag_smoothing.items()})

        for lag in (feature_lags if isinstance(feature_lags, Iterable) else range(feature_lags)):
            # if smoothed values are applicable use smoothed values
            if smoothers is not None and len(smoothers) > 0 and smoothers.peekitem(0)[0] <= lag:
                feature_series = smoothers.popitem(0)[1]

            # assign the lagged (eventually smoothed) feature to the features frame
            dff[(feature, lag)] = feature_series.shift(lag)

    if multi_index:
        # fix tuple column index to actually be a multi index and fix levels
        # features need to be row, time step, feature to fit RNN based models
        dff.columns = pd.MultiIndex.from_tuples(dff.columns).swaplevel(0, 1)

    return dff