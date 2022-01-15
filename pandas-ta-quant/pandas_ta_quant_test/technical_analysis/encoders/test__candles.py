from unittest import TestCase

import pandas as pd

from pandas_ta_quant import np
from pandas_ta_quant.technical_analysis.encoders import ta_candles_as_culb
from pandas_ta_quant_test.config import DF_TEST


class TestCandleEncoder(TestCase):

    def test_multi_channel(self):
        df = DF_TEST.copy()
        ohlc_values = df[["Open", "High", "Low", "Close"]].iloc[-1].values
        culb = ta_candles_as_culb(df, volume=None)
        culb_values = culb.iloc[-1].values

        self.assertGreaterEqual(culb["upper"].values.max(), 0)
        self.assertGreaterEqual(culb["lower"].values.max(), 0)
        self.assertEqual(ohlc_values[1] > ohlc_values[-1], culb_values[-1] < 0)
        np.testing.assert_array_almost_equal(np.array([3.120900e+02, 6.402011e-05, 4.861506e-03, -4.48435e-04]),
                                             culb_values, 5)

    def test_culb_inf(self):
        df = pd.DataFrame({
            "Open": [0, 0, 1],
            "High": [0, 0, 1],
            "Low": [0, 0, 1],
            "Close": [0, 0, 1],
            "Volume": [0, 0, 1],
        })

        culb = ta_candles_as_culb(df, relative='log')
        self.assertTrue(culb["body"].replace([np.inf, -np.inf], np.nan).isna().any())
        self.assertFalse(culb["Volume"].replace([np.inf, -np.inf], np.nan).isna().any())
