import pandas as pd
import numpy as np

from pandas_ml_utils_test.ml.model.test_model_accuracy import TestModelAccuracy


class TestSkModelAccuracy(TestModelAccuracy):

    def provide_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        from pandas_ml_utils import FeaturesAndLabels, SkModel
        return SkModel(LinearRegression(), FeaturesAndLabels(["x"], ["y"]))


class TestSkModelAccuracyMLP(TestModelAccuracy):

    def provide_linear_regression_model(self):
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import FeaturesAndLabels, SkModel

        return SkModel(
            MLPRegressor(10, activation='identity', learning_rate_init=0.1, max_iter=8000, validation_fraction=0),
            FeaturesAndLabels(["x"], ["y"])
        )

    def provide_non_linear_regression_model(self):
        from sklearn.neural_network import MLPRegressor
        from pandas_ml_utils import FeaturesAndLabels, SkModel

        return SkModel(
            MLPRegressor(200, learning_rate_init=0.001, max_iter=5000, validation_fraction=0),
            FeaturesAndLabels(["x"], ["y"])
        )

