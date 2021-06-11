from functools import partial
from unittest import TestCase

from scipy.stats import norm
from sklearn.neural_network import MLPRegressor

from pandas_ml_utils import SkModel, FeaturesAndLabels, FittingParameter
from pandas_ml_utils.ml.callback import EarlyStopping, TestConfidenceInterval
from pandas_ml_utils.ml.confidence import CdfConfidenceInterval
from pandas_ml_utils_test.config import DF_AB_10
import pandas as pd
import numpy as np


class TestCallBack(TestCase):

    def test_early_stopping(self):
        """given an early stopping callback"""
        cb = EarlyStopping(2, tolerance=-10)

        """when training is not improveing"""
        with DF_AB_10.model() as m:
            m.fit(
                SkModel(
                    MLPRegressor([1, 2, 1]),
                    FeaturesAndLabels(["a"], ["b"]),
                ),
                FittingParameter(epochs=50),
                verbose=1,
                callbacks=cb
            )

        """then not all epochs were executed"""
        self.assertLess(cb.call_counter, 50)
        self.assertGreaterEqual(cb.call_counter, cb.patience)

    def test_confidence(self):
        """given a confidence interval test callback"""
        df = pd.DataFrame({"a": np.random.normal(0, 0.3, 1500), "b": np.random.normal(0, 0.5, 1500)})
        cb = TestConfidenceInterval(
            CdfConfidenceInterval(lambda param, val: norm.cdf(val, loc=0, scale=param), 0.8, expand_args=True),
            lambda row: row
        )

        """when training"""
        with df.model() as m:
            m.fit(
                SkModel(
                    MLPRegressor([1, 20, 1], activation='tanh'),
                    FeaturesAndLabels(["a"], ["b"]),
                ),
                FittingParameter(epochs=10),
                callbacks=cb
            )

        """then cb was executed"""
        self.assertGreaterEqual(cb.call_counter, 0)
        self.assertGreaterEqual(len(cb.history), 1)
