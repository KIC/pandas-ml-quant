from unittest import TestCase

import numpy as np
import torch.nn as nn

from pandas_ml_quant_rl.model.agent import PolicyNetwork


class TestPyTochNetwork(TestCase):

    def test_tupple_unzipping(self):
        numbers = np.random.random((10,))
        net = PolicyNetwork()
        net.foo = nn.Linear(1, 1)

        nested = net._unzip_to_tensor(((numbers, numbers), numbers))
        regular = net._unzip_to_tensor((numbers, numbers))

