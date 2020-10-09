from unittest import TestCase
import pandas as pd
from pandas_ml_utils import ClassificationSummary, RegressionSummary
import numpy as np
from pandas_ml_utils.constants import *


class TestSummary(TestCase):

    def test_classification_summary(self):
        df = pd.DataFrame({"a": np.hstack([np.zeros(5), np.ones(5)]), "b": np.random.random(10)})
        df.columns = pd.MultiIndex.from_product([[LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME], ["a"]])

        html = ClassificationSummary(df, None)._repr_html_()

        # render html http://htmledit.squarefree.com/
        print(html)

        self.assertIn("data:image/png;base64", html)

    def test_regression_summary(self):
        df = pd.DataFrame({"a": np.random.random(10), "b": np.random.random(10)})
        df.columns = pd.MultiIndex.from_product([[LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME], ["a"]])

        html = RegressionSummary(df, None)._repr_html_()

        # render html http://htmledit.squarefree.com/
        print(html)

        self.assertIn("data:image/png;base64", html)

