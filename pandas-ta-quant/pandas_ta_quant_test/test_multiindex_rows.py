from unittest import TestCase

from pandas_ta_quant import pd
from pandas_ta_quant_test.config import DF_TEST


class TestMultiIndexRowDf(TestCase):

    def test_indicator(self):
        df1 = DF_TEST.copy()
        df2 = DF_TEST.copy()
        df1.index = pd.MultiIndex.from_product([["A"], df1.index.tolist()])
        df2.index = pd.MultiIndex.from_product([["B"], df2.index.tolist()])
        df = pd.concat([df1, df2], axis=0)

        self.assertEqual(df.loc["A"].shape, df.loc["B"].shape)

        res = df["Close"].ta.macd()
        pd.testing.assert_frame_equal(res.loc["A"], res.loc["B"])
