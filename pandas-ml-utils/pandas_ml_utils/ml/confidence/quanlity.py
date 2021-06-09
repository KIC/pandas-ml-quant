from typing import Union, Tuple, Iterable, Callable

import numpy as np
import pandas as pd


class CdfConfidenceInterval(object):

    def __init__(self,
                 cdf_provider: Callable[[pd.DataFrame], float],
                 interval: Union[float, Tuple[float, float]] = 0.95,
                 expand_args=False):
        self.cdf_provider = cdf_provider
        # get left and right tail threshold i.e. 0.025, 0.975
        self.left_confidence, self.right_confidence = \
            interval if isinstance(interval, Iterable) else ((1. - interval) / 2, (1. - interval) / 2 + interval)
        self.max_tail_events = self.left_confidence + (1. - self.right_confidence)
        self.expand_args = expand_args

    def apply(self, df: pd.DataFrame, *args, **kwargs):
        assert df.ndim == 2 and df.shape[1] == 2, "Expected a dataframe with columns=(distribution_parameters, value)"

        probs = df.apply(self, axis=1)
        tail_events = ((probs < self.left_confidence).sum() + (probs > self.right_confidence).sum())
        if hasattr(tail_events, 'item'):
            tail_events = tail_events.item()

        return tail_events / float(len(df))

    def __call__(self, row: pd.Series, *args, **kwargs):
        args = (row.iloc[0], row.iloc[1]) if not isinstance(row, np.ndarray) else (row[0], row[1])
        return self.cdf_provider(*args) if self.expand_args else self.cdf_provider(args)
