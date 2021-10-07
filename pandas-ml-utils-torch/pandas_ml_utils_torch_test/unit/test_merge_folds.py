from unittest import TestCase
from torch.optim import Adam
from torch import nn
import torch

from pandas_ml_utils_torch.merging_cross_folds import average_folds
from pandas_ml_utils_torch.pytorch_base import PytochBaseModel, PytorchNN


class TestMergeFolds(TestCase):

    def test_average_merge(self):
        class Net(PytorchNN):

            def __init__(self, init_weight):
                super().__init__()
                self.l1 = nn.Linear(1, 1)
                torch.nn.init.constant_(self.l1.weight, init_weight)

            def forward(self, x):
                return self.l1(x)


        a = PytochBaseModel(lambda: Net(1), nn.MSELoss, Adam)
        b = PytochBaseModel(lambda: Net(2), nn.MSELoss, Adam)

        res = list(average_folds({1: a, 2: b}).net.parameters())
        self.assertAlmostEqual(res[0].data.numpy().item(), 1.5)
