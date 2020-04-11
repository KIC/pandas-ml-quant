from unittest import TestCase

from pandas_ml_quant.analysis import ta_garch11
from pandas_ml_quant_test.config import DF_TEST
import numpy as np


class TestVolatilityForecast(TestCase):

    def test_garch11(self):
        df = DF_TEST[-220:].copy()

        garch11 = ta_garch11(df[["Close"]])

        print(garch11)
        np.testing.assert_almost_equal(garch11.iloc[-1, -1], 0.000040, 5)

