from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import SGD

from pandas_ml_utils import pd, AutoEncoderModel, FeaturesAndLabels
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel, Callbacks
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


class AutoEncoderModule(nn.Module):

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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        with t.no_grad():
            return self.encoder(t.from_numpy(x).float()).numpy()

    def decode(self, x):
        with t.no_grad():
            return self.decoder(t.from_numpy(x).float()).numpy()


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

    def provide_auto_encoder_model(self, features_and_labels):
        t.manual_seed(12)

        model = AutoEncoderModel(
            PytorchModel(
                features_and_labels,
                AutoEncoderModule,
                nn.MSELoss,
                lambda params: SGD(params, lr=0.1, momentum=0.9)
            ),
            ["condensed"],
            lambda m: m.module.encode,
            lambda m: m.module.decode,
        )

        return model

    def test_callbacks(self):
        # a test with a early stopping callback and pass restore_best_weights=True as kwarg
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0, ],
            "b": [0, 1, 0, 1, 1, 0, 1, 0, ],
        })

        model = PytorchModel(
            FeaturesAndLabels(["a", "b"], ["b"]),
            ClassificationModule,
            nn.MSELoss,
            lambda params: SGD(params, lr=0.1, momentum=0.9)
        )

        fit = df.model.fit(model, on_epoch=[Callbacks.early_stopping(patience=3, tolerance=-100)], restore_best_weights=True)
        self.assertEqual(4, len(fit.model.history["loss"]))
