from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_utils.ml.summary.figures import df_regression_scores
from pandas_ml_utils_test.config import DF_SUMMARY


class TestFigures(TestCase):

    def test_regression_figures(self):
        print(df_regression_scores(DF_SUMMARY, None))
        self.assertFalse(df_regression_scores(DF_SUMMARY, None).isnull().values.any())
