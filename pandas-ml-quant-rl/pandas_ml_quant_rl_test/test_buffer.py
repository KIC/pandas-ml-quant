from unittest import TestCase
import numpy as np


from pandas_ml_quant_rl.buffer.list_buffer import ListBuffer


class TestBuffers(object):

    def init_buffer(self, buffer, ignore_capacity=False):
        for i in range(9):
            buffer.append_row(np.array([i, 10 - i]))

        buffer.append_row(np.array([9, 0]))

        return buffer


class TestListBuffer(TestBuffers, TestCase):

    def test_init(self):
        self.init_buffer(ListBuffer(2), True)


