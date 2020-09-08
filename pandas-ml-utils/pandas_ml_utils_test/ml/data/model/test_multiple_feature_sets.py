from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_utils import pd
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel
from pandas_ml_utils import FeaturesAndLabels


class TestMultiFeatureSet(TestCase):

    def test_pytorch_mfs(self):
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0,],
            "b": [[0, 0], [0, 0], [1, 1], [1,1], [0,0], [0,0], [1,1], [1,1],],
            "c": [1, 0, 0, 1, 1, 0, 0, 1,]
        })

        def module_provider():
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

            return ClassificationModule()

        model = PytorchModel(
            FeaturesAndLabels(
                features=(["a"], ["b"]),
                labels=["c"]
            ),
            module_provider,
            nn.MSELoss,
            lambda params: Adam(params, lr=0.03)
        )

        (f, _), l, t, s, g = df._.extract(model.features_and_labels)
        self.assertIsInstance(f, MultiFrameDecorator)
        print(f)

#        fit = df.model.fit(model)
        # FIXME print(fit)
