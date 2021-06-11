from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.stats import norm

from pandas_ml_utils.ml.confidence import NormalConfidence, CdfConfidenceInterval


class TestConfidence(TestCase):

    def test_df_confidence_interval(self):
        std = 0.3
        conf = 0.75
        df = pd.DataFrame({"a": np.random.normal(0, std, 2000)})
        x = pd.Series(np.zeros(len(df)), index=df.index).to_frame().apply(partial(NormalConfidence(conf), std=std), result_type='expand', axis=1)
        tail_events = ((df["a"] >= x[0]) & (df["a"] <= x[1])).values.sum()
        tail_events = tail_events / len(df)
        self.assertGreaterEqual(tail_events, conf - 0.015)

    def test_df_confidence_fit(self):
        std = 0.3
        conf = 0.75

        cdf = CdfConfidenceInterval(lambda param, val: norm.cdf(val, loc=0, scale=param), conf, expand_args=True)
        df = pd.DataFrame({"a": np.random.normal(0, std, 2000)})
        df["std"] = std

        # print(df[["std", "a"]].apply(cdf, axis=1))
        # print(cdf.apply(df[["std", "a"]]))
        quality = cdf.apply(df[["std", "a"]])
        self.assertLessEqual(quality, conf + 0.015)
        self.assertGreaterEqual(quality, 1 - conf - 0.015)
