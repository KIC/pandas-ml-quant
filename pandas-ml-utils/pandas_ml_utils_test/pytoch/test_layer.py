import torch as t
from unittest import TestCase

from pandas_ml_utils.pytorch.layers import KerasLikeLSTM


class TestPytorchLayer(TestCase):

    def test_lstm_layer(self):
        data = t.rand((1, 10, 2))
        lstm = KerasLikeLSTM((10, 2), 2, return_sequence=False)
        lstm2 = KerasLikeLSTM((10, 2), 2, return_sequence=True)

        self.assertEqual((1, 2), lstm(data).shape)
        self.assertEqual((1, 10, 2), lstm2(data).shape)

