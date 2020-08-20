from unittest import TestCase
import numpy as np


from pandas_ml_quant_rl.buffer.array_buffer import ArrayBuffer
from pandas_ml_quant_rl.buffer.list_buffer import ListBuffer


class TestBuffers(object):

    def init_buffer(self, buffer, ignore_capacity=False):
        for i in range(9):
            buffer.add(np.array([i, 10 - i]))
            assert not buffer.is_full or ignore_capacity

        buffer.add(np.array([9, 0]))
        assert buffer.is_full or ignore_capacity

        return buffer

    def rank_and_cut(self, buffer):
        ranked = buffer.rank_and_cut((1, 0.5))
        assert len(ranked.data) == 5
        assert ranked.data[:, 1].sum() == 40, ranked.data[:, 1].sum()


class TestArrayBuffer(TestBuffers, TestCase):

    def test_init(self):
        self.init_buffer(ArrayBuffer((10, 2)))

    def test_rank_and_cut(self):
        self.rank_and_cut(self.init_buffer(ArrayBuffer((10, 2))))


class TestListBuffer(TestBuffers, TestCase):

    def test_init(self):
        self.init_buffer(ListBuffer(2), True)

    def test_rank_and_cut(self):
        self.rank_and_cut(self.init_buffer(ListBuffer(2), True))


