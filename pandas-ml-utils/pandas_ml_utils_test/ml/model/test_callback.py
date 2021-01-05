from unittest import TestCase

from sklearn.neural_network import MLPRegressor

from pandas_ml_utils import SkModel, FeaturesAndLabels, FittingParameter
from pandas_ml_utils.ml.callback import EarlyStopping
from pandas_ml_utils_test.config import DF_AB_10


class TestCallBack(TestCase):

    def test_early_stopping(self):
        """given early stopping callback"""
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