from unittest import TestCase

import numpy as np
import torch as T
from torch import nn
from torch.distributions import MixtureSameFamily, Categorical, Normal
from torch.optim import Adam

from pandas_ml_common import naive_splitter, FeaturesLabels
from pandas_ml_utils import pd, FittingParameter
from pandas_ml_utils.ml.callback import TestConfidenceInterval
from pandas_ml_utils.ml.model.base_model import FittableModel
from pandas_ml_utils_torch import PytorchModelProvider, PytorchNNFactory
from pandas_ml_utils_torch import layers
from pandas_ml_utils_torch import lossfunction
from pandas_ml_utils_torch.utils import wrap_applyable
from pandas_ml_quant.model.summary.price_prediction_summary import PriceSampledSummary


class TestPytorchProbabilisticModel(TestCase):

    def test_probabilistic_model_with_callback(self):
        df = pd.DataFrame({"Returns": np.random.normal(-0.02, 0.03, 500) + np.random.normal(0.03, 0.02, 500)})

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

        def get_cdf(*args):
            params, prediction = args[0]
            return cdf_cb(T.tensor(params).reshape(1, -1)).cdf(T.tensor(prediction)).numpy()

        def get_variance(*args):
            params, prediction = args[0].values
            return cdf_cb(T.tensor(params).reshape(1, -1)).variance

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
            FeaturesLabels(
                features=["Returns"],
                features_postprocessor=lambda df: df.ta.rnn(20),
                labels=[lambda df: df["Returns"].shift(-1).rename("Future_Returns")],
                reconstruction_targets=lambda df: (1 + df["Returns"]).cumprod().rename("Close")
            ),
            summary_provider=summary_provider
        )

        fit = df.model.fit(
            model,
            FittingParameter(epochs=10, batch_size=6, splitter=naive_splitter(0.25)),
            # verbose=1,
            callbacks=[
                TestConfidenceInterval(
                    TestConfidenceInterval.CdfConfidenceInterval(get_cdf, interval=0.8),
                    get_variance,
                    early_stopping=True
                )
            ]
        )

        print(fit.test_summary.calc_scores())
        self.assertIsNotNone(fit.test_summary._repr_html_())

    def test_summary_provider(self):
        summary_provider = PriceSampledSummary.with_reconstructor(
            label_returns=lambda y: y,
            label_reconstruction=lambda target: target.iloc[:, 0],
            sampler=wrap_applyable(
                lambda params, samples: Normal(params[..., 0], params[..., 1]).sample([int(samples.item())]),
                nr_args=2),
            forecast_period=7,
            samples=500,
            confidence=0.7,
        ),
