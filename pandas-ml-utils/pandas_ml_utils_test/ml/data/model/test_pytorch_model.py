from unittest import TestCase

import torch.nn as nn
from torch.optim import SGD

from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel


class TestKerasModel(TestAbstractModel, TestCase):

    def provide_regression_model(self):

        class RegressionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.regressor = nn.Sequential(
                    nn.Linear(1, 1)
                )

            def forward(self, x):
                x = self.regressor(x)
                return x

        model = PytorchModel(
            FeaturesAndLabels(features=["a"], labels=["b"]),
            RegressionModule,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.01, momentum=0.0)
        )

        return model


