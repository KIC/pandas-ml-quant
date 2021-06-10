from unittest import TestCase

import torch as t

from pandas_ml_utils_torch.distribution import Logistic


class TestDistribution(TestCase):

    def test_logistic(self):
        dist = Logistic(t.tensor([0.0] * 3), t.tensor([1.0] * 3))
        print(dist.sample())