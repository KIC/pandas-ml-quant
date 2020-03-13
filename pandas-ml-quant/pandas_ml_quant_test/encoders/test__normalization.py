from unittest import TestCase
from pandas_ml_quant import pd, np
from pandas_ml_quant.encoders import ta_rescale, ta_realative_candles
from pandas_ml_quant_test.config import DF_TEST


class TestRescaling(TestCase):

    def test_rescale(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 4, 5],
                           "b": [5, 4, 3, 2, 2, 2]})

        series = ta_rescale(df["a"])
        all = ta_rescale(df)
        columns = ta_rescale(df, axis=0)
        rows = ta_rescale(df, axis=1)

        self.assertEqual(1, series.values[-1])
        self.assertListEqual([1, -0.5], all[-1:].values[-1].tolist())
        self.assertListEqual([1, -1], columns[-1:].values[-1].tolist())
        self.assertListEqual([1, -1], rows[-1:].values[-1].tolist())

    def test_relative_candles(self):
        df = DF_TEST[-2:].copy()
        relative = ta_realative_candles(df)

        np.testing.assert_array_almost_equal(np.array([0.002469,  0.002021,  0.002533, -0.002829]),
                                             relative.values[-1])
