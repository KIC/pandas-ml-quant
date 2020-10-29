from unittest import TestCase

from sklearn.neural_network import MLPRegressor

from pandas_ml_utils import SkModel, FeaturesAndLabels
from pandas_ml_utils.ml.model.callback import EarlyStopping
from pandas_ml_utils_test.config import DF_AB_10


class TestCallBack(TestCase):

    def test_early_stopping(self):
        """given early stopping callback"""
        cb = EarlyStopping(2)

        """when training is not improveing"""
        with DF_AB_10.model() as m:
            m.fit(
                SkModel(
                    MLPRegressor([1, 2, 1]),
                    FeaturesAndLabels(["a"], ["b"]),
                ),
                fold_epochs=50,
                verbose=1,
                callbacks=cb
            )

        """then not all epochs were executed"""
        self.assertLess(cb.call_conter, 50)