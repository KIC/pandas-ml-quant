import importlib
from unittest import TestCase

import pandas as pd

from pandas_ml_common import naive_splitter
from pandas_ml_utils import PostProcessedFeaturesAndLabels, FittingParameter
from pandas_ml_utils.ml.callback import TestConfidenceInterval
from pandas_ml_utils_torch import PytorchModel, PytorchNNFactory, LambdaSplitter
from pandas_ml_utils_torch.loss import DistributionNLL
from torch.optim import Adam
from torch.distributions import MixtureSameFamily, Categorical, Normal
from torch import nn
import torch as T
import numpy as np

from pandas_ml_utils_torch.utils import wrap_applyable


class TestProbabilisticModel(TestCase):

    def test_probabilistic_model_with_callback(self):
        try:
            pandas_ml_quant_data_provider = importlib.import_module("pandas_ml_quant")
        except:
            print("pandas_ml_quant not found, skipping!")
            return

        df = pd.DataFrame({"Returns": np.random.normal(-0.02, 0.03, 500) + np.random.normal(0.03, 0.02, 500)})

        fl = PostProcessedFeaturesAndLabels(
            features=["Returns"],
            feature_post_processor=lambda df: df.ta.rnn(20),
            labels=[
                lambda df: df["Returns"].shift(-1).rename("Future_Returns")
            ]
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

        model = PytorchModel(
            module_provider=model_factory,
            features_and_labels=fl,
            criterion_provider=lambda: DistributionNLL(loss),
            optimizer_provider=Adam,
            restore_best_weights=True,
        )

        fit = df.model.fit(
            model,
            FittingParameter(epochs=50, batch_size=6, splitter=naive_splitter(0.25)),
            #verbose=1,
            callbacks=[
                TestConfidenceInterval(
                    TestConfidenceInterval.CdfConfidenceInterval(
                        lambda *args: lambda x: dist(T.Tensor(args[:2]), T.Tensor(args[2:4]), T.Tensor(args[4:6])).cdf(T.Tensor([x]))
                    ),
                    early_stopping=True
                )
            ]
        )
