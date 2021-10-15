from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

from pandas_ml_common import FeaturesLabels
from pandas_ml_utils import FittableModel, SkModelProvider
from pandas_ml_utils.ml.summary.figures import df_regression_scores, df_classification_scores, df_feature_importance
from pandas_ml_utils_test.config import DF_SUMMARY_REGRESSION, DF_SUMMARY_CLASSIFICATION, DF_NOTES


class TestFigures(TestCase):

    def test_regression_figures(self):
        print(df_regression_scores(DF_SUMMARY_REGRESSION, no_style=True))
        self.assertFalse(df_regression_scores(DF_SUMMARY_REGRESSION, no_style=True).isnull().values.any())

    def test_classification_figures(self):
        print(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None, no_style=True))
        #print(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None)._repr_html_())
        self.assertNotEqual(df_classification_scores(DF_SUMMARY_CLASSIFICATION, None, no_style=True).max().item(), np.nan)

    def test_feature_importance(self):
        with DF_NOTES.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(
                        MLPRegressor(activation='tanh')
                    ),
                    FeaturesLabels(
                        features=["variance", "skewness", "kurtosis", "entropy"],
                        labels="authentic",
                        label_type=float
                    )
                )
            )

        ranked_features = df_feature_importance(fit.test_frames, fit.model, verbose=True)
        print(ranked_features)

        self.assertEqual(
            ["entropy", "kurtosis", "skewness", "variance"][-2:],
            ranked_features.index.tolist()[-2:]
        )
