from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_multi_ma, ta_naive_edge_detect, ta_edge_detect_mean, \
    ta_edge_detect_poly
from pandas_ta_quant_test.config import DF_TEST
import numpy as np


class TestFilter(TestCase):

    def test_multi_ma(self):
        df = DF_TEST
        mma = ta_multi_ma(df)

        self.assertEqual(len(mma.columns), 30)

    def test_naive_edge(self):
        edges1 = ta_naive_edge_detect(DF_TEST["Close"])
        edges2 = ta_edge_detect_mean(DF_TEST["Close"])
        edges3 = ta_edge_detect_poly(DF_TEST["Close"])
        print(DF_TEST[["Close"]].join(edges1).join(edges2).join(edges3).tail(20))

