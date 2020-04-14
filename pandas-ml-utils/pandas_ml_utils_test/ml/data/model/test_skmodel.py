from unittest import TestCase

from sklearn.neural_network import MLPRegressor, MLPClassifier

from pandas_ml_common import np, pd
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils import SkModel, FeaturesAndLabels


class TestSkModel(TestAbstractModel, TestCase):

    def provide_classification_model(self, features_and_labels):
        model = SkModel(
            MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001, solver='lbfgs', random_state=42),
            features_and_labels,
        )

        return model

    def provide_regression_model(self, features_and_labels):
        model = SkModel(
            MLPRegressor(1, learning_rate_init=0.01, solver='sgd', activation='identity', momentum=0, max_iter=1500, n_iter_no_change=500, nesterovs_momentum=False, shuffle=False, validation_fraction=0.0, random_state=42),
            features_and_labels,
        )

        return model