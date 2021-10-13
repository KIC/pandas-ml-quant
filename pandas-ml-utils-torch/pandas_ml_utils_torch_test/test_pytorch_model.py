import importlib
from typing import Tuple, Dict, Any
from unittest import TestCase

import numpy as np
import torch as T
import torch as t
from torch import nn
from torch.distributions import MixtureSameFamily, Categorical, Normal
from torch.optim import Adam
from torch.optim import SGD

from pandas_ml_common import naive_splitter, FeaturesLabels
from pandas_ml_utils import pd, FittingParameter
from pandas_ml_utils.ml.callback import TestConfidenceInterval
from pandas_ml_utils.ml.model.base_model import ModelProvider, FittableModel
from pandas_ml_utils_test.ml.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils_torch import PytorchNN, PytorchModelProvider, PytorchNNFactory, PytorchAutoEncoderFactory
from pandas_ml_utils_torch import layers
from pandas_ml_utils_torch import lossfunction
from pandas_ml_utils_torch.utils import wrap_applyable


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

    def test_probabilistic_model_with_callback(self):
        try:
            pandas_ml_quant_data_provider = importlib.import_module("pandas_ml_quant")
            from pandas_ml_quant import PricePredictionSummary
            from pandas_ml_quant.model.summary.price_prediction_summary import PriceSampledSummary
        except:
            print("pandas_ml_quant not found, skipping!")
            return

        df = pd.DataFrame({"Returns": np.random.normal(-0.02, 0.03, 500) + np.random.normal(0.03, 0.02, 500)})

        fl = FeaturesLabels(
            features=["Returns"],
            features_postprocessor=lambda df: df.ta.rnn(20),
            labels=[
                lambda df: df["Returns"].shift(-1).rename("Future_Returns")
            ],
            reconstruction_targets=lambda df: (1 + df["Returns"]).cumprod().rename("Close")
        )

        model_factory = PytorchNNFactory.create(
            nn.Sequential(
                nn.Linear(20, 10),
                nn.Tanh(),
                nn.Linear(10, 6),
                layers.LambdaSplitter(
                    lambda x: T.softmax(x[..., :2], dim=1),
                    lambda x: T.exp(x[..., 2:4]),
                    # enforce one mean positive and the other negativ
                    lambda x: T.cat([T.exp(x[..., 4:5]), -T.exp(x[..., 5:6])], dim=1),
                )
            ),
            predictor=lambda n, i: T.cat(n(i), dim=1),
            trainer=lambda n, i: n(i)
        )

        def dist(probs, scales, locs):
            return MixtureSameFamily(
                Categorical(probs=probs),
                Normal(loc=locs, scale=scales)
            )

        def loss(y_pred):
            probs, scales, locs = y_pred
            return dist(probs, scales, locs)

        def cdf_cb(arg):
            probs, scales, locs = arg[..., :2], arg[..., 2:4], arg[..., 4:6]
            return dist(probs, scales, locs)

        summary_provider = PriceSampledSummary.with_reconstructor(
            sampler=wrap_applyable(lambda params, samples: cdf_cb(params).sample([int(samples.item())]), nr_args=2),
            samples=100,
            confidence=0.8
        )

        model = FittableModel(
            PytorchModelProvider(
                model_factory,
                lambda: lossfunction.DistributionNLL(loss, penalize_toal_variance_lambda=1.1),
                Adam,
            ),
            fl,
            summary_provider=summary_provider
        )

        fit = df.model.fit(
            model,
            FittingParameter(epochs=10, batch_size=6, splitter=naive_splitter(0.25)),
            #verbose=1,
            callbacks=[  # FIXME fix failing callback
                TestConfidenceInterval(
                    TestConfidenceInterval.CdfConfidenceInterval(
                        wrap_applyable(lambda params, val: cdf_cb(params).cdf(val), nr_args=2),
                        interval=0.8
                    ),
                    wrap_applyable(lambda params: cdf_cb(params).variance),
                    early_stopping=True
                )
            ]
        )

        print(fit.test_summary.calc_scores())


