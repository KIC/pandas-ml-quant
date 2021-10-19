from typing import Tuple, Dict, Any
from unittest import TestCase

import torch as t
from torch import nn
from torch.optim import Adam
from torch.optim import SGD

from pandas_ml_common import naive_splitter, FeaturesLabels
from pandas_ml_utils import pd, FittingParameter
from pandas_ml_utils.ml.model.base_model import ModelProvider, FittableModel
from pandas_ml_utils_test.ml.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils_torch import PytorchNN, PytorchModelProvider, PytorchNNFactory, PytorchAutoEncoderFactory


class TestPytorchModel(TestAbstractModel, TestCase):

    def test_multi_sample_regressor(self):
        super().test_multi_sample_regressor()

    def test_no_test_data(self):
        super().test_no_test_data()

    def test_multindex_row(self):
        super().test_multindex_row()

    def test_multindex_row_multi_samples(self):
        super().test_multindex_row_multi_samples()

    def test_stacked_models(self):
        super().test_stacked_models()

    def test_concatenated_multi_models(self):
        super().test_concatenated_multi_models()

    def test_classifier(self):
        super().test_classifier()

    def test_regressor(self):
        super().test_regressor()

    def test_auto_encoder(self):
        super().test_auto_encoder()

    def provide_regression_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        return (
            PytorchModelProvider(
                PytorchNNFactory.create(
                    nn.Sequential(nn.Linear(1, 1)),
                ),
                nn.MSELoss,
                lambda params: SGD(params, lr=0.03)
            ),
            dict(batch_size=None, epochs=500)
        )

    def provide_classification_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        t.manual_seed(42)

        return (
            PytorchModelProvider(
                PytorchNNFactory.create(
                    nn.Sequential(
                        nn.Linear(2, 5),
                        nn.ReLU(),
                        nn.Linear(5, 1),
                        nn.Sigmoid()
                    ),
                ),
                nn.MSELoss,
                lambda params: SGD(params, lr=0.03)
            ),
            dict(batch_size=None, epochs=500)
        )

    def provide_auto_encoder_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        t.manual_seed(12)

        return (
            PytorchModelProvider(
                PytorchAutoEncoderFactory(
                    encoder=nn.Sequential(
                        nn.Linear(2, 2),
                        nn.Tanh(),
                        nn.Linear(2, 1),
                        nn.Tanh(),
                    ),
                    decoder=nn.Sequential(
                        nn.Linear(1, 2),
                        nn.Tanh(),
                        nn.Linear(2, 2),
                        nn.Tanh(),
                    )
                ),
                nn.MSELoss,
                lambda params: SGD(params, lr=0.1, momentum=0.9),
                is_auto_encoder=True
            ),
            dict(batch_size=None, epochs=500)
        )

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
                FittableModel(
                    PytorchModelProvider(NN, nn.MSELoss, Adam),
                    FeaturesLabels(features=["a"], labels=["b"]),
                ),
                FittingParameter(
                    splitter=naive_splitter(0.5),
                    epochs=2,
                    fold_epochs=10,
                    batch_size=2
                )
            )

        print(fit)

