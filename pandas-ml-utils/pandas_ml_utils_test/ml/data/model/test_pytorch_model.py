from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import SGD

from pandas_ml_utils.ml.model.pytoch_model import PytorchModel
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel


class ClassificationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class RegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class TestPytorchModel(TestAbstractModel, TestCase):

    def provide_classification_model(self, features_and_labels):
        t.manual_seed(42)

        model = PytorchModel(
            features_and_labels,
            ClassificationModule,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.03)
        )

        return model

    def provide_regression_model(self, features_and_labels):
        t.manual_seed(12)

        model = PytorchModel(
            features_and_labels,
            RegressionModule,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.01, momentum=0.0)
        )

        return model


