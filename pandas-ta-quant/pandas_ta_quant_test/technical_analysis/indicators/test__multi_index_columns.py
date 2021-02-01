from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ta_quant.technical_analysis import ta_candles_as_culb
from pandas_ta_quant_test.config import DF_TEST_MULTI


class TestMultiIndexColumn(TestCase):

    def test_culbs(self):
        df = DF_TEST_MULTI.copy()
        multi = ta_candles_as_culb(df)
        self.assertEqual(5*2, multi.shape[1])
