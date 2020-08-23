from unittest import TestCase

import numpy as np
import torch.nn as nn

from pandas_ml_quant_rl.model.agent import PolicyNetwork


class TestPyTochNetwork(TestCase):

    def test_tupple_unzipping(self):
        numbers = np.random.random((10,))
        net = PolicyNetwork()
        net.foo = nn.Linear(1, 1)

        regular = net._unzip_to_tensor([(numbers, numbers)])
        print(regular[0].shape, regular[1].shape)

        batch = [(n, n) for n in numbers]
        regular = net._unzip_to_tensor(batch)
        print(regular[0].shape, regular[1].shape)

        batch = [((n, n), n) for n in numbers]
        nested = net._unzip_to_tensor(batch)
        print(nested[0][0].shape, nested[0][1].shape, nested[1].shape)

        self.assertEqual(10, regular[0].shape[0])
        self.assertEqual(10, regular[1].shape[0])

        self.assertEqual(10, nested[0][0].shape[0])
        self.assertEqual(10, nested[0][1].shape[0])
        self.assertEqual(10, nested[1].shape[0])

