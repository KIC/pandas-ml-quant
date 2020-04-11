from unittest import TestCase
from pandas_ml_quant import pd
from pandas_ml_quant.analysis.labels import ta_opening_gap, ta_future_bband_quantile, ta_future_multi_bband_quantile
from pandas_ml_quant_test.config import DF_TEST


class TestDescreteLabels(TestCase):

    def test_ta_opening_gap(self):
        df = pd.DataFrame({"Open": [0, 2, 2, 2],
                           "Close":   [1, 2, 3, 0]})

        label = ta_opening_gap(df)

        self.assertListEqual([2., 0., 1.], label.dropna().tolist())

    def test_ta_future_bband_quantile(self):
        df = DF_TEST.copy()

        label1 = ta_future_bband_quantile(df["Close"], include_mean=False).dropna()
        label2 = ta_future_bband_quantile(df["Close"], include_mean=True).dropna()

        self.assertEqual(2.0, label1.values.max())
        self.assertEqual(3.0, label2.values.max())

    def test_ta_future_multi_bband_quantile(self):
        df = DF_TEST.copy()

        label = ta_future_multi_bband_quantile(df["Close"], include_mean=False).dropna()

        self.assertEqual(0.0, label.values.min())
        self.assertEqual(8.0, label.values.max())


