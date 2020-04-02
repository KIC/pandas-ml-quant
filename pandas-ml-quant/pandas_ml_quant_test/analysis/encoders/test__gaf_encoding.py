from unittest import TestCase
from pandas_ml_quant import np
from pandas_ml_quant.analysis.encoders import ta_gaf, ta_rnn
from pandas_ml_quant_test.config import DF_TEST


class TestGAF(TestCase):

    def test_series(self):
        s = DF_TEST["Close"]
        timesteps = ta_rnn(s, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf._.values.shape
        np.testing.assert_almost_equal(0.37, gaf.iloc[-1][0][0][0], 2)
        self.assertEqual((6760, 1, 3, 3), shape)

    def test_multi_channel(self):
        df = DF_TEST[["Open", "Close"]]
        timesteps = ta_rnn(df, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf._.values.shape
        self.assertEqual((6760, 2, 3, 3), shape)
