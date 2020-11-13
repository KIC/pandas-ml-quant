from unittest import TestCase

from pandas_ml_quant import _TA
from pandas_ml_quant_test.config import DF_TEST_MULTI, DF_TEST_MULTI_ROW, DF_TEST_MULTI_ROW_MULTI_COLUMN
import numpy as np


class TestTheAllIndicator(TestCase):

    def test_all_indicator_multi_columns(self):
        mdf = DF_TEST_MULTI.copy()

        # test does not throw any exception
        # TechnicalAnalysis(df).all()
        res = _TA(mdf).all()

        print(res.tail())
        print(res.columns)
        self.assertEqual(3, res.columns.nlevels)
        self.assertEqual(3835, len(res))

    def test_all_indicator_multirows(self):
        mdf = DF_TEST_MULTI_ROW_MULTI_COLUMN.copy()

        # test does not throw any exception
        # TechnicalAnalysis(df).all()
        res = _TA(mdf).all()

        print(res.tail())
        print(res.columns)
        self.assertEqual(3, res.columns.nlevels)
        self.assertEqual(2, res.index.nlevels)
        self.assertEqual(3835 * 2, len(res))
        np.testing.assert_array_almost_equal(res.loc["A"].values, res.loc["B"], 5)

