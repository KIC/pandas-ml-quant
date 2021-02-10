from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_utils.ml.confidence import NormalConfidence


class TestNormalConfidence(TestCase):

    def test_df_confidence(self):
        std = 0.3
        conf = 0.75
        df = pd.DataFrame({"a": np.random.normal(0, std, 1500)})
        x = pd.Series(np.zeros(len(df)), index=df.index).to_frame().apply(partial(NormalConfidence(conf), std=std), result_type='expand', axis=1)
        tail_events = ((df["a"] >= x[0]) & (df["a"] <= x[1])).values.sum()
        tail_events = tail_events / len(df)
        self.assertGreaterEqual(tail_events, conf - 0.02)
