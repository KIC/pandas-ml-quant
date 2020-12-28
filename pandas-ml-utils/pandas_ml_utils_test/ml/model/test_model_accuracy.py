from unittest import TestCase

from sklearn.model_selection import KFold

from pandas_ml_common import dummy_splitter
from pandas_ml_utils.constants import *
import pandas as pd
import numpy as np


def create_line_data(n=300, slope=1):
    np.random.seed(32)
    x = np.linspace(0, 1, n)
    y = slope * x + np.random.normal(0, 0.05, n)
    return x, y


def create_sine_data(n=300):
    np.random.seed(32)
    x = np.linspace(0, 1 * 2 * np.pi, n)
    y1 = 3 * np.sin(x)
    y1 = np.concatenate((np.zeros(60), y1 + np.random.normal(0, 0.15 * np.abs(y1), n), np.zeros(60)))
    x = np.concatenate((np.linspace(-3, 0, 60), np.linspace(0, 3 * 2 * np.pi, n),
                        np.linspace(3 * 2 * np.pi, 3 * 2 * np.pi + 3, 60)))
    y2 = 0.1 * x + 1
    y = (y1 + y2) + 2
    return x, y


class TestModelAccuracy(TestCase):

    def test_linear_regression(self):
        model = self.provide_linear_regression_model()
        if model is None:
            return

        df = pd.DataFrame(np.array(create_line_data(300)).T, columns=["x", "y"])
        with df.model() as m:
            fit = m.fit(model, splitter=dummy_splitter)
            y = fit.training_summary.df[LABEL_COLUMN_NAME].values
            y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
            dist = np.sqrt(np.sum((y - y_hat) ** 2))

        print(dist)
        self.assertLessEqual(dist, 0.87)

    def test_linear_regression_kfold(self):
        model = self.provide_linear_regression_model()
        if model is None:
            return

        df = pd.DataFrame(np.array(create_line_data(300)).T, columns=["x", "y"])
        with df.model() as m:
            fit = m.fit(model, splitter=dummy_splitter, cross_validation=KFold(2))
            y = fit.training_summary.df[LABEL_COLUMN_NAME].values
            y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
            dist = np.sqrt(np.sum((y - y_hat) ** 2))

        print(dist)
        self.assertLessEqual(dist, 1.07)

    def test_non_linear_regression(self):
        model = self.provide_non_linear_regression_model()
        if model is None:
            return

        df = pd.DataFrame(np.array(create_sine_data(300)).T, columns=["x", "y"])
        with df.model() as m:
            fit = m.fit(model, splitter=dummy_splitter)
            y = fit.training_summary.df[LABEL_COLUMN_NAME].values
            y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
            dist = np.sqrt(np.sum((y - y_hat) ** 2))

        print(dist)
        self.assertLessEqual(dist, 9)

    def provide_linear_regression_model(self):
        return None

    def provide_non_linear_regression_model(self):
        return None

