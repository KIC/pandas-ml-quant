from unittest import TestCase

import numpy as np
import torch as t
import torch.nn as nn

from pandas_ml_quant.pytorch.layers import Time2Vec
from pandas_ml_quant_test.config import DF_TEST


class TestTime2Vec(TestCase):

    def test_time2vec(self):
        df = DF_TEST.copy()

        x = df["Volume"].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        class MyModule(nn.Module):

            def __init__(self, input, output):
                super().__init__()
                self.net = Time2Vec(input, output)

            def forward(self, x):
                return self.net(x)

        t2v = MyModule(5, 32)
        print(x.shape)
        res = t2v(t.from_numpy(x).float())
        print(res)
        print(res.shape)  # -> ?, 5, 33

    def _test_lala(self):
        class MyModule(nn.Module):

            def __init__(self, input, output):
                super().__init__()
                self.net = Time2Vec(input, output)

            def forward(self, x):
                return self.net(x)

        t2v = MyModule(3, 3)
        print(t2v(t.from_numpy(np.array([[[0.1], [0.2], [0.3]], [[0.1], [0.2], [0.3]]])).float()))