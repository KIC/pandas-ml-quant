import os
import tempfile
import uuid
from unittest import TestCase

import numpy as np
import torch as t
from torch import nn
from torch.optim import Adam

from pandas_ml_common import serialize, deserialize
from pandas_ml_utils_torch.pytorch_base import PytochBaseModel, PytorchNN, PytorchNNFactory


class ANet(PytorchNN):

    def __init__(self, something):
        super().__init__()
        self.net = nn.Linear(1, 1, False)
        print(something)

    def forward_training(self, *input) -> t.Tensor:
        return self.net(input[0])


class TestPytorchBaseModel(TestCase):

    def test_linreg_fit(self):
        x = t.from_numpy(np.linspace(0, 1, 300).reshape(-1, 1)).float()
        y = t.from_numpy((np.linspace(0, 1, 300) + np.random.normal(0, 0.05, 300)).reshape(-1, 1)).float()

        m = PytochBaseModel(
            PytorchNNFactory.create(
                nn.Sequential(nn.Linear(1, 1, False)),
                lambda net, x: net(x)
            ),
            nn.MSELoss,
            Adam
        )

        err = [m.fit_epoch(x, y, None) for _ in range(3000)]
        self.assertLess(err[-1], err[0])

        self.assertAlmostEqual(((m.predict(x, numpy=False) - y) ** 2).mean().sqrt().cpu().item(), 0.05, 1)
        # print(m.predict(x).numpy())
        # print(y.numpy())

        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        try:
            serialize(m, temp)
            self.assertLess(((m.predict(x, numpy=False) - deserialize(temp, PytochBaseModel).predict(x, numpy=False)) ** 2).mean().cpu().item(), 1e-5)
        finally:
            os.remove(temp)

    def test_non_linear_fit(self):
        def create_sine_data(n=300):
            np.random.seed(32)
            x = np.linspace(0, 1 * 2 * np.pi, n)
            y1 = 3 * np.sin(x)
            y1 = np.concatenate((np.zeros(60), y1 + np.random.normal(0, 0.15 * np.abs(y1), n), np.zeros(60)))
            x = np.concatenate((np.linspace(-3, 0, 60), np.linspace(0, 3 * 2 * np.pi, n),
                                np.linspace(3 * 2 * np.pi, 3 * 2 * np.pi + 3, 60)))
            y2 = 0.1 * x + 1
            y = (y1 + y2) + 2
            return t.from_numpy(x).reshape(-1, 1).float(), t.from_numpy(y).reshape(-1, 1).float()

        t.manual_seed(0)

        class Net(PytorchNN):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 200),
                    nn.ReLU(),
                    nn.Linear(200, 200),
                    nn.ReLU(),
                    nn.Linear(200, 1),
                    nn.ReLU(),
                )

            def forward_training(self, *input) -> t.Tensor:
                return self.net(input[0])

        m = PytochBaseModel(
            Net,
            nn.MSELoss,
            Adam
        )

        x, y = create_sine_data()
        err = [m.fit_epoch(x, y, None) for _ in range(3000)]
        self.assertLess(err[-1], err[0])

        dist = ((m.predict(x, numpy=False) - y) ** 2).mean().sqrt().cpu().item()
        self.assertLess(dist, 0.5)
        # print(dist)
        # print("x = np." + repr(y.numpy()).replace(', dtype=float32', ''))
        # print("y = np." + repr(m.predict(x).numpy()).replace(', dtype=float32', ''))
