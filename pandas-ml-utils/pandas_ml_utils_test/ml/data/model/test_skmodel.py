from unittest import TestCase

from sklearn.neural_network import MLPRegressor

from pandas_ml_common import np, pd
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils import SkModel, FeaturesAndLabels


class TestSkModel(TestAbstractModel, TestCase):

    def provide_regression_model(self):
        model = SkModel(
            MLPRegressor(1, learning_rate_init=0.01, solver='sgd', activation='identity', momentum=0, max_iter=1500, n_iter_no_change=500, nesterovs_momentum=False, shuffle=False, validation_fraction=0.0, random_state=42),
            FeaturesAndLabels(features=["a"], labels=["b"]),
        )

        return model