import importlib
from unittest import TestCase

import numpy as np
import pandas as pd
import torch as T
from torch import nn
from torch.distributions import MixtureSameFamily, Categorical, Normal
from torch.optim import Adam

from pandas_ml_common import naive_splitter
from pandas_ml_utils import PostProcessedFeaturesAndLabels, FittingParameter
from pandas_ml_utils.ml.callback import TestConfidenceInterval
from pandas_ml_utils_torch import PytorchModel, PytorchNNFactory, LambdaSplitter
from pandas_ml_utils_torch.loss import DistributionNLL
from pandas_ml_utils_torch.utils import wrap_applyable


class TestProbabilisticModel(TestCase):

    def test_probabilistic_model_with_callback(self):
        try:
            pandas_ml_quant_data_provider = importlib.import_module("pandas_ml_quant")
            from pandas_ml_quant import PricePredictionSummary
            from pandas_ml_quant.model.summary.price_prediction_summary import PriceSampledSummary
        except:
            print("pandas_ml_quant not found, skipping!")
            return

        df = pd.DataFrame({"Returns": np.random.normal(-0.02, 0.03, 500) + np.random.normal(0.03, 0.02, 500)})

        fl = PostProcessedFeaturesAndLabels(
            features=["Returns"],
            feature_post_processor=lambda df: df.ta.rnn(20),
            labels=[
                lambda df: df["Returns"].shift(-1).rename("Future_Returns")
            ],
            targets=lambda df: (1 + df["Returns"]).cumprod().rename("Close")
        )

        model_factory = PytorchNNFactory.create(
            nn.Sequential(
                nn.Linear(20, 10),
                nn.Tanh(),
                nn.Linear(10, 6),
                LambdaSplitter(
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

        model = PytorchModel(
            module_provider=model_factory,
            features_and_labels=fl,
            criterion_provider=lambda: DistributionNLL(loss, penalize_toal_variance_lambda=1.1),
            optimizer_provider=Adam,
            summary_provider=summary_provider
        )


        fit = df.model.fit(
            model,
            FittingParameter(epochs=10, batch_size=6, splitter=naive_splitter(0.25)),
            #verbose=1,
            callbacks=[
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
