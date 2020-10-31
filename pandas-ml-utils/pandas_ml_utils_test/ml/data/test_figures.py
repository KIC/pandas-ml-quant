from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_utils.ml.summary.figures import df_regression_scores, df_classification_scores
from pandas_ml_utils_test.config import DF_SUMMARY_REGRESSION, DF_SUMMARY_CLASSIFICATION


class TestFigures(TestCase):

    def test_regression_figures(self):
        print(df_regression_scores(DF_SUMMARY_REGRESSION, None, no_style=True))
        self.assertFalse(df_regression_scores(DF_SUMMARY_REGRESSION, None, no_style=True).isnull().values.any())

    def test_classification_figures(self):
        print(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None, no_style=True))
        #print(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None)._repr_html_())
        self.assertNotEqual(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None, no_style=True).max().item(), np.nan)

