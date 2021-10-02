from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor

from pandas_ml_utils import SkModelProvider, FeaturesLabels, FittingParameter, FittableModel
from pandas_ml_utils.ml.callback import EarlyStopping, TestConfidenceInterval, NbLiveLossPlot
from pandas_ml_utils.ml.confidence import CdfConfidenceInterval
from pandas_ml_utils_test.config import DF_AB_10


class TestCallBack(TestCase):

    def test_live_loss(self):
        """given an early stopping callback"""
        cb = NbLiveLossPlot(backend='svg')

        """when training is not improveing"""
        with DF_AB_10.model() as m:
            m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor([1, 2, 1])),
                    FeaturesLabels(features=["a"], labels=["b"]),
                ),
                FittingParameter(epochs=10),
                verbose=1,
                callbacks=cb
            )

        """then no error is thrown"""
        pass

    def test_early_stopping(self):
        """given an early stopping callback"""
        cb = EarlyStopping(2, tolerance=-10)

        """when training is not improveing"""
        with DF_AB_10.model() as m:
            m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor([1, 2, 1])),
                    FeaturesLabels(features=["a"], labels=["b"]),
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
            lambda row: row.iloc[0][0]
        )

        """when training"""
        with df.model() as m:
            m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor([1, 20, 1], activation='relu')),
                    FeaturesLabels(features=["a"], labels=["b"]),
                ),
                FittingParameter(epochs=10),
                callbacks=cb
            )

        """then cb was executed"""
        self.assertGreaterEqual(cb.call_counter, 0)
        self.assertGreaterEqual(len(cb.history), 1)
