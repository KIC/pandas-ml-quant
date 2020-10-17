from unittest import TestCase
import pandas as pd
import numpy as np

from pandas_ml_common import Sampler
from pandas_ml_quant.model.cross_validation.rolling_window_cv import RollingWindowCV


class TestCrossValidation(TestCase):

    def test_rolling_cv(self):
        df = pd.DataFrame({"feature": np.arange(10), "label": np.arange(10)})
        sampler = Sampler(df[["feature"]], df[["label"]], splitter=None, cross_validation=RollingWindowCV(5, 2))
        samples = list(sampler.sample_cross_validation())
        first_fold, last_fold = samples[0], samples[-2]

        self.assertEqual(0, first_fold[2][1].iloc[0, 0])
        self.assertEqual(2, first_fold[3][1].iloc[0, 0])

        self.assertEqual(9, last_fold[3][1].iloc[-1, 0])
        self.assertEqual(7, last_fold[2][1].iloc[-1, 0])
