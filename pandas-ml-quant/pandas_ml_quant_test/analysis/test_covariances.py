from unittest import TestCase

from pandas_ml_quant import pd
from pandas_ml_quant.technichal_analysis import ta_ewma_covariance
from pandas_ml_quant_test.config import DF_TEST_MULTI_ROW, DF_TEST_MULTI


class TestCovariances(TestCase):

    def test_ewma_covariance(self):
        df = DF_TEST_MULTI._["Close"]
        cov = ta_ewma_covariance(df)
        print(cov.tail())

    def test_ewma_covariance_multiindex_row(self):
        df = pd.concat([DF_TEST_MULTI.add_multi_index("A", axis=0), DF_TEST_MULTI.add_multi_index("B", axis=0)], axis=0)
        cov = ta_ewma_covariance(df)
        pd.testing.assert_frame_equal(cov.loc["A"], cov.loc["B"])
