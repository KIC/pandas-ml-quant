from unittest import TestCase

import os
import torch.nn as nn
from torch.optim import Adam
from pandas_ml_common import pd
from pandas_ml_utils import KerasModel, AutoEncoderModel, FeaturesAndLabels
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestKerasModel(TestCase):

    def test_classifier(self):
        # test safe and load
        pass

    def test_regressor(self):
        df = pd.DataFrame({
            "a": [0.2, 0.4, 0.6, 0.8],
            "b": [0.1, 0.2, 0.3, 0.4]
        })

        class RegressionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.regressor = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.Linear(1, 1),
                )

            def forward(self, x):
                x = self.regressor(x)
                return x

        model = PytorchModel(
            FeaturesAndLabels(features=["a"], labels=["b"]),
            RegressionModule,
            nn.MSELoss,
            Adam
        )

        fit = df.model.fit(model)
        print(fit.test_summary.df)

        prediction = df.model.predict(fit.model)
        print(prediction)
        # TODO test save load model same result

    def test_custom_object(self):
        # test safe and load
        pass


