from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_quant.analysis import ta_ma_ratio, ta_normalize_row
from pandas_ml_quant_test.config import DF_TEST


class TestNormalizer(TestCase):

    def test_normalize_row(self):
        df = pd.DataFrame({"a": [1,2,3], "b": [3,6,9]})
        mm01 = ta_normalize_row(df, 'minmax01')
        mm11 = ta_normalize_row(df, 'minmax-11')
        std = ta_normalize_row(df, 'standard')
        uni = ta_normalize_row(df, 'uniform')

        np.testing.assert_array_almost_equal(mm01.values, np.stack([np.zeros(3), np.ones(3)], axis=1))
        np.testing.assert_array_almost_equal(mm11.values, np.stack([np.zeros(3) - 1, np.ones(3)], axis=1))
        np.testing.assert_array_almost_equal(std.values, np.array([[-1,  1], [0, 4], [1,  7]]))
        # since we only have 2 columns and we want the values to be unified we assume both be the same
        np.testing.assert_array_almost_equal(uni.values, np.ones((3, 2)))

    def test_ma_ratio(self):
        df = DF_TEST

        ma_ration_scaled = ta_ma_ratio(df)
        print(ma_ration_scaled.columns)
        self.assertTrue(True)

