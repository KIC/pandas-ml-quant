from unittest import TestCase
from pandas_ml_quant.df.technical_analysis import TechnicalAnalysis
from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI


class TestTheAllIndicator(TestCase):

    def test_all_indicator(self):
        # df = DF_TEST.copy()
        mdf = DF_TEST_MULTI.copy()

        # test does not throw any exception
        # TechnicalAnalysis(df).all()
        res = TechnicalAnalysis(mdf).all()

        print(res.tail())
        print(res.columns)
        self.assertEqual(3835, len(res))



