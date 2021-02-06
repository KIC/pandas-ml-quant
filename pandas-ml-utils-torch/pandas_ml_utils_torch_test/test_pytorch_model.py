from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import SGD, Adam

from pandas_ml_common import naive_splitter
from pandas_ml_utils import pd, FeaturesAndLabels, FittingParameter
from pandas_ml_utils.ml.model.base_model import AutoEncoderModel
from pandas_ml_utils_test.ml.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils_torch import PytorchModel, PytorchAutoEncoderModel, PytorchNN


class RegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x, **kwargs):
        x = self.regressor(x)
        return x


class AutoEncoderModule(PytorchNN):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
        )

    def forward_training(self, x, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ClassificationModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        x = self.classifier(x)
        return x


class TestPytorchModel(TestAbstractModel, TestCase):

    def provide_batch_size_and_epoch(self):
        return None, 500

    def provide_classification_model(self, features_and_labels):
        t.manual_seed(42)

        model = PytorchModel(
            ClassificationModule,
            features_and_labels,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.03)
        )

        return model

    def provide_regression_model(self, features_and_labels):
        t.manual_seed(12)

        model = PytorchModel(
            RegressionModule,
            features_and_labels,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.01, momentum=0.0)
        )

        return model

    def provide_auto_encoder_model(self, features_and_labels):
        t.manual_seed(12)

        model = PytorchAutoEncoderModel(
            AutoEncoderModule,
            features_and_labels,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.1, momentum=0.9)
        )

        return model

    def test_mult_epoch_cross_validation(self):
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0, ],
            "b": [0, 1, 0, 1, 1, 0, 1, 0, ],
        })

        with df.model() as m:
            class NN(PytorchNN):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.nn = nn.Sequential(
                        nn.Linear(1, 2),
                        nn.ReLU(),
                        nn.Linear(2, 1),
                    )

                def forward_training(self, x):
                    return self.nn(x)

            fit = m.fit(
                PytorchModel(NN, FeaturesAndLabels(["a"], ["b"]), nn.MSELoss, Adam),
                FittingParameter(
                    splitter=naive_splitter(0.5),
                    epochs=2,
                    fold_epochs=10,
                    batch_size=2
                )
            )

        print(fit)