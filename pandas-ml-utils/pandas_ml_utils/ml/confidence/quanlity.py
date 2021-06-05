from typing import Union, Tuple, Iterable, Callable

import pandas as pd


class CdfConfidenceInterval(object):

    def __init__(self,
                 cdf_provider: Callable[[pd.Series], Callable[[float], float]],
                 interval: Union[float, Tuple[float, float]] = 0.95,
                 expand_args=False):
        self.cdf_provider = cdf_provider
        self.expand_args = expand_args
        # get left and right tail threshold i.e. 0.025, 0.975
        self.left_confidence, self.right_confidence = \
            interval if isinstance(interval, Iterable) else ((1. - interval) / 2, (1. - interval) / 2 + interval)
        self.max_tail_events = self.left_confidence + (1. - self.right_confidence)

    def apply(self, df: pd.DataFrame, *args, **kwargs):
        assert df.ndim == 2 and df.shape[1] == 2, "Expected a dataframe with columns=(distribution_parameters, value)"

        probs = df.apply(self, axis=1)
        tail_events = ((probs < self.left_confidence).sum() + (probs > self.right_confidence).sum())
        if hasattr(tail_events, 'item'):
            tail_events = tail_events.item()

        return tail_events / float(len(df))

    def __call__(self, row: pd.Series, *args, **kwargs):
        if self.expand_args:
            args = row.iloc[0] if isinstance(row.iloc[0], Iterable) else [row.iloc[0]]
            prob = self.cdf_provider(*args)(row.iloc[1])
        else:
            prob = self.cdf_provider(row.iloc[0])(row.iloc[1])

        return prob
