from unittest import TestCase

from pandas_ta_quant import pd
from pandas_ta_quant.technical_analysis import ta_ewma_covariance, ta_mgarch_covariance
from pandas_ta_quant_test.config import DF_TEST_MULTI_ROW, DF_TEST_MULTI


class TestCovariances(TestCase):

    def test_ewma_covariance(self):
        df = DF_TEST_MULTI._["Close"]
        cov = ta_ewma_covariance(df)
        # print(cov.tail())

    def test_ewma_covariance_multiindex_row(self):
        df = pd.concat([DF_TEST_MULTI.add_multi_index("A", axis=0), DF_TEST_MULTI.add_multi_index("B", axis=0)], axis=0)
        cov = ta_ewma_covariance(df)
        pd.testing.assert_frame_equal(cov.loc["A"], cov.loc["B"])

    def test_mgarch_covariance(self):
        df = DF_TEST_MULTI[-40:]._["Close"]
        cov = ta_mgarch_covariance(df)
        # print(cov)

        # ewma cov result:
        #.00002107488e-05 -1.341822e-05
        #  (Close, gld) -1.341822e-05  2.034593e-05
        self.assertGreater(cov.iloc[-2, -2], 2.107488e-05)
        self.assertLess(cov.iloc[-1, -2], 0)
