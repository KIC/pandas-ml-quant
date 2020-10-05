from unittest import TestCase

import torch as t
import torch.nn as nn

from pandas_ml_common import np
from pandas_ml_common.utils.column_lagging_utils import lag_columns
from pandas_ml_utils_test.config import DF_TEST
from pandas_ml_utils_torch.layers import KerasLikeLSTM, RegularizedLayer, Time2Vec


class TestPytorchLayer(TestCase):

    def test_lstm_layer(self):
        data = t.rand((1, 10, 2))
        lstm = KerasLikeLSTM((10, 2), 2, return_sequence=False)
        lstm2 = KerasLikeLSTM((10, 2), 2, return_sequence=True)

        self.assertEqual((1, 2), lstm(data).shape)
        self.assertEqual((1, 10, 2), lstm2(data).shape)

    def test_regularizer_wrapper(self):
        m = nn.Sequential(
            nn.Linear(10, 2),
            nn.ReLU(),
            RegularizedLayer(nn.Linear(2, 1), 0.01),
            nn.ReLU(),
            nn.Linear(2, 55, bias=False),
        )

        # print(list(m.parameters()))
        regularized = [p for p in m.parameters() if hasattr(p, "_regularizers")]
        print(regularized)

        self.assertEqual(len(list(m.parameters())), 5)
        self.assertEqual(len(regularized), 1)

    def test_time2vec(self):
        df = DF_TEST.copy()

        x = lag_columns(df["Volume"].pct_change(), 5).dropna()._.values[:-2]
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