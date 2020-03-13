from unittest import TestCase
from pandas_ml_quant import pd, np
from pandas_ml_quant.encoders import ta_gaf
from pandas_ml_quant.indicators import ta_shape_for_auto_regression
from pandas_ml_quant_test.config import DF_TEST


class TestGAF(TestCase):

    def test_series(self):
        s = DF_TEST["Close"]
        timesteps = ta_shape_for_auto_regression(s, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf.ml.values.shape
        np.testing.assert_almost_equal(0.370, gaf.iloc[-1][0][0], 0.001)
        self.assertEqual((6760, 1, 3, 3), shape)

    def test_multi_channel(self):
        df = DF_TEST[["Open", "Close"]]
        timesteps = ta_shape_for_auto_regression(df, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf.ml.values.shape
        self.assertEqual((6760, 2, 3, 3), shape)
