from unittest import TestCase

import torch as t

from pandas_ml_utils_torch import distribution


class TestDistribution(TestCase):

    def test_logistic(self):
        dist = distribution.Logistic(t.tensor([0.0] * 3), t.tensor([1.0] * 3))
        print(dist.sample())