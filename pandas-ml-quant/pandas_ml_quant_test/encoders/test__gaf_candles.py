from unittest import TestCase

from pandas_ml_quant import np
from pandas_ml_quant.encoders import ta_candles_as_culb
from pandas_ml_quant_test.config import DF_TEST


class TestCandleEncoder(TestCase):

    def test_multi_channel(self):
        df = DF_TEST.copy()
        club = ta_candles_as_culb(df)

        self.assertGreaterEqual(club["upper"].values.max(), 0)
        self.assertGreaterEqual(club["lower"].values.max(), 0)
        np.testing.assert_array_almost_equal(np.array([3.120900e+02, 6.402011e-05, 4.861506e-03, 9.995516e-01]),
                                             club.iloc[-1].values, 5)
