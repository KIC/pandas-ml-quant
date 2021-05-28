from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ta_quant._decorators import for_each_top_level_column, is_time_consuming
from pandas_ta_quant._utils import conditional_func
from pandas_ta_quant.technical_analysis import ta_cci, ta_adx, ta_vola_hurst
from pandas_ta_quant_test.config import DF_TEST_MULTI


class TestAnalysisUtils(TestCase):

    def test_conditional_func(self):
        s = pd.Series(np.linspace(0, 1, 10))
        res = conditional_func(s >= 0.5, s * -10, s * 10)

        print(res)

    def test_decorators(self):
        """given a multi index data frame"""
        df = DF_TEST_MULTI

        """when executing an indicator"""
        func = for_each_top_level_column(ta_cci)
        res = func(df)

        func2 = for_each_top_level_column(ta_adx)
        res2 = func2(df)

        """then we got a column for each top level column"""
        self.assertEqual(2, res.columns.nlevels)
        self.assertEqual(2, res2.columns.nlevels)
        self.assertEqual(10, len(res2.columns))


    def test_has_decorator(self):
        self.assertTrue(_test_has_dec_true._is_timeconsuming)
        self.assertTrue(ta_vola_hurst._is_timeconsuming)
        self.assertFalse(hasattr(_test_has_dec_false, '_is_timeconsuming'))
        self.assertEqual(_test_has_dec_true(), 12)
        self.assertEqual(_test_has_dec_false(), 12)


def _test_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_decorated = True
    return wrapper


@is_time_consuming
def _test_has_dec_true():
    return 12


def _test_has_dec_false():
    return 12
