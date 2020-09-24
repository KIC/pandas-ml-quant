from unittest import TestCase

import torch as t
import torch.nn as nn
from pandas_ml_utils.pytorch.layers import KerasLikeLSTM, RegularizedLayer


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
