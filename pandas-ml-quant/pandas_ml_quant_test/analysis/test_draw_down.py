from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_quant.technichal_analysis import ta_draw_down
from pandas_ml_quant_test.config import DF_TEST

pd.set_option('display.max_columns', None)


class TestDrawDown(TestCase):

    def test_draw_down(self):
        df = DF_TEST.copy()
        dd = ta_draw_down(df["Close"], return_dates=True, return_duration=True)

        self.assertAlmostEqual(-0.035417, dd.loc["1993-03-02"]["mdd"], 6)
        self.assertListEqual(
            [45.125000, 0.000000, 0.000000, pd.Timestamp("1993-02-04"), pd.Timestamp("1993-03-03"), 18.0],
            dd.loc["1993-03-03"].to_list()
        )
