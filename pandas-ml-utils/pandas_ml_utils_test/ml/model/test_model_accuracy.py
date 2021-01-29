from typing import Tuple, List
from unittest import TestCase

from sklearn.model_selection import KFold

from pandas_ml_common import dummy_splitter
from pandas_ml_utils import Model, FittingParameter
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
    return x / x[-1], y


class TestModelAccuracy(TestCase):

    def test_linear_regression(self):
        models = self.provide_linear_regression_model()
        if models is None:
            return

        df = pd.DataFrame(np.array(create_line_data(300)).T, columns=["x", "y"])
        for model, fp in  models:
            with df.model() as m:
                fit = m.fit(model, fp.with_splitter(splitter=dummy_splitter).with_cross_validation(None))
                y = fit.training_summary.df[LABEL_COLUMN_NAME].values
                y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
                dist = np.sqrt(np.sum((y - y_hat) ** 2))

            print(fp.context, dist)
            losses = list(fit.model._history[("train", 0)].values())

            self.assertLessEqual(dist, 1.22, fp.context)
            if len(losses) > 1:
                self.assertLess(losses[-1], losses[0])

    def test_linear_regression_kfold(self):
        models = self.provide_linear_regression_model()
        if models is None:
            return

        df = pd.DataFrame(np.array(create_line_data(300)).T, columns=["x", "y"])
        for model, fp in models:
            with df.model() as m:
                fit = m.fit(model, fp.with_splitter(splitter=dummy_splitter).with_cross_validation(KFold(4)))
                y = fit.training_summary.df[LABEL_COLUMN_NAME].values
                y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
                dist = np.sqrt(np.sum((y - y_hat) ** 2))

            print(fp.context, dist)
            self.assertLessEqual(dist, 1.6, fp.context)

    def test_non_linear_regression(self):
        models = self.provide_non_linear_regression_model()
        if models is None:
            return

        df = pd.DataFrame(np.array(create_sine_data(300)).T, columns=["x", "y"])
        for model, fp in models:
            with df.model() as m:
                fit = m.fit(model, fp.with_splitter(splitter=dummy_splitter).with_cross_validation(None))
                y = fit.training_summary.df[LABEL_COLUMN_NAME].values
                y_hat = fit.training_summary.df[PREDICTION_COLUMN_NAME].values
                dist = np.sqrt(np.sum((y - y_hat) ** 2))

            print(fp.context, dist, len(fit.model._history[("train", 0)]))
            self.assertLessEqual(dist, 10, fp.context)

    def provide_linear_regression_model(self) -> List[Tuple[Model, FittingParameter]]:
        return None

    def provide_non_linear_regression_model(self) -> List[Tuple[Model, FittingParameter]]:
        return None

