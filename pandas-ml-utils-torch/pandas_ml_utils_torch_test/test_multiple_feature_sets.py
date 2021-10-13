import unittest
from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels
from pandas_ml_utils import pd, FittingParameter
from pandas_ml_utils_torch import PytorchModelProvider, PytorchNN
from pandas_ml_utils import FeaturesLabels, FittableModel
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

                def forward_training(self, x0, x1) -> t.Tensor:
                    return self.net0(x0) + self.net1(x1)

            return ClassificationModule()

        model = FittableModel(
            PytorchModelProvider(
                module_provider,
                nn.MSELoss,
                lambda params: Adam(params, lr=0.03)
            ),
            FeaturesLabels(
                features=[["a"], ["b"]],
                labels=["c"]
            ),
        )

        fl: FeaturesWithLabels = df.ML.extract(model.features_and_labels_definition).extract_features_labels_weights()
        self.assertIsInstance(fl.features_with_required_samples.features, list)
        print(fl.features_with_required_samples.features)

        fit = df.model.fit(model, FittingParameter(fold_epochs=10))
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
                FeaturesLabels(
                    features=[
                        ["a", "b"],
                        ["b", "a"]
                    ],
                    features_postprocessor=[
                        lambda df: df + 1,
                        lambda df: df + 2,
                    ],
                    labels=["c"]
                )
            ).extract_features_labels_weights()

        self.assertEqual(ext.features[0].sum(axis=1).sum(), 24)
        self.assertEqual(ext.features[1].sum(axis=1).sum(), 40)
