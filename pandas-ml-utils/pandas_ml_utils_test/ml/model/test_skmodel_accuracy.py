from unittest import TestCase

from pandas_ml_utils import FittingParameter
from pandas_ml_utils_test.ml.model.test_model_accuracy import TestModelAccuracy


class TestSkModelAccuracy(TestModelAccuracy, TestCase):

    def test_linear_regression(self):
        super().test_linear_regression()

    def test_linear_regression_kfold(self):
        super().test_linear_regression_kfold()

    def test_non_linear_regression(self):
        super().test_non_linear_regression()

    def provide_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import SkModelProvider

        return [
            (
                SkModelProvider(LinearRegression()),
                FittingParameter(epochs=1, fold_epochs=1, context="LinearRegression")
            ),
            (
                SkModelProvider(MLPRegressor(10, learning_rate_init=0.01, max_iter=9000, validation_fraction=0)),
                FittingParameter(epochs=1, fold_epochs=1, context="MLPRegressor")
            ),
            (
                SkModelProvider(MLPRegressor(10, learning_rate_init=0.01, max_iter=1, validation_fraction=0, warm_start=True)),
                FittingParameter(epochs=9000, fold_epochs=1, context="MLPRegressor partial fit")
            )
        ]

    def provide_non_linear_regression_model(self):
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import SkModelProvider

        return [
            (
                SkModelProvider(MLPRegressor(200, learning_rate_init=0.001, max_iter=5000, validation_fraction=0)),
                FittingParameter(epochs=1, context="epoch 1 fit"),
            ),
            (
                SkModelProvider(MLPRegressor(200, learning_rate_init=0.001, max_iter=1, validation_fraction=0, warm_start=True)),
                FittingParameter(epochs=5000, context="partial fit"),
            )
        ]



