import unittest
from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_utils import pd
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels
from pandas_ml_utils_torch import PytorchModel, PytorchNN
from pandas_ml_utils import FeaturesAndLabels, PostProcessedFeaturesAndLabels
from pandas_ml_utils.constants import FEATURE_COLUMN_NAME
import torch as t
import numpy as np


class TestMultiFeatureSet(TestCase):

    def test_pytorch_mfs(self):
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0,],
            "b": [[0, 0], [0, 0], [1, 1], [1,1], [0,0], [0,0], [1,1], [1,1],],
            "c": [1, 0, 0, 1, 1, 0, 0, 1,]
        })

        def module_provider():
            class ClassificationModule(PytorchNN):

                def __init__(self):
                    super().__init__()
                    self.net0 = nn.Sequential(
                        nn.Linear(1, 5),
                        nn.ReLU(),
                        nn.Linear(5, 1),
                        nn.Sigmoid()
                    )
                    self.net1 = nn.Sequential(
                        nn.Linear(2, 5),
                        nn.ReLU(),
                        nn.Linear(5, 1),
                        nn.Sigmoid()
                    )

                def forward_training(self, x) -> t.Tensor:
                    x0, x1 = x
                    return self.net0(x0) + self.net1(x1)

            return ClassificationModule()

        model = PytorchModel(
            module_provider,
            FeaturesAndLabels(
                features=(["a"], ["b"]),
                labels=["c"]
            ),
            nn.MSELoss,
            lambda params: Adam(params, lr=0.03)
        )

        fl: FeaturesWithLabels = df._.extract(model.features_and_labels)
        self.assertIsInstance(fl.features_with_required_samples.features, MultiFrameDecorator)
        print(fl.features_with_required_samples.features)

        fit = df.model.fit(model, fold_epochs=10)
        print(fit.test_summary.df)

        self.assertIn(FEATURE_COLUMN_NAME, fit.test_summary.df)
        np.testing.assert_almost_equal(np.array([0, 0, 1]), fit.test_summary.df["label"].values.squeeze())

    def test_postprocessed_multiple_featue_sets(self):
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0, ],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, ],
            "c": [1, 0, 0, 1, 1, 0, 0, 1, ]
        })

        with df.model() as m:
            ext = m.extract(
                PostProcessedFeaturesAndLabels(
                    features=(
                        ["a", "b"],
                        ["b", "a"]
                    ),
                    feature_post_processor=(
                        [lambda df: df + 1],
                        [lambda df: df + 2],
                    ),
                    labels=["c"]
                )
            )

        self.assertEqual(ext.features.frames()[0].sum(axis=1).sum(), 24)
        self.assertEqual(ext.features.frames()[1].sum(axis=1).sum(), 40)
