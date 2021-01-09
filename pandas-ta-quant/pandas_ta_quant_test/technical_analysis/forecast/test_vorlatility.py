from unittest import TestCase

import numpy as np

from pandas_ta_quant.technical_analysis import ta_garch11
from pandas_ta_quant_test.config import DF_TEST


class TestVolatilityForecast(TestCase):

    def test_garch11(self):
        df = DF_TEST[-220:].copy()

        garch11 = ta_garch11(df[["Close"]])

        print(garch11)
        np.testing.assert_almost_equal(garch11.iloc[-1, -1], 0.000040, 5)

