from pandas_ml_utils import FittingParameter
from pandas_ml_utils_test.ml.model.test_model_accuracy import TestModelAccuracy


class TestSkModelAccuracy(TestModelAccuracy):

    def provide_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import FeaturesAndLabels, SkModel

        return [
            (
                SkModel(LinearRegression(), FeaturesAndLabels(["x"], ["y"])),
                FittingParameter(epochs=1, fold_epochs=1, context="LinearRegression")
            ),
            (
                SkModel(
                    MLPRegressor(10, learning_rate_init=0.01, max_iter=9000, validation_fraction=0),
                    FeaturesAndLabels(["x"], ["y"])
                 ),
                FittingParameter(epochs=1, fold_epochs=1, context="MLPRegressor")
            ),
            (
                SkModel(
                    MLPRegressor(10, learning_rate_init=0.01, max_iter=1, validation_fraction=0, warm_start=True),
                    FeaturesAndLabels(["x"], ["y"])
                 ),
                FittingParameter(epochs=9000, fold_epochs=1, context="MLPRegressor partial fit")
            )
        ]

    def provide_non_linear_regression_model(self):
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import FeaturesAndLabels, SkModel

        return [
            (
                SkModel(
                    MLPRegressor(200, learning_rate_init=0.001, max_iter=5000, validation_fraction=0),
                    FeaturesAndLabels(["x"], ["y"])
                ),
                FittingParameter(epochs=1, context="epoch 1 fit"),
            ),
            (
                SkModel(
                    MLPRegressor(200, learning_rate_init=0.001, max_iter=1, validation_fraction=0, warm_start=True),
                    FeaturesAndLabels(["x"], ["y"])
                ),
                FittingParameter(epochs=5000, context="partial fit"),
            )
        ]



