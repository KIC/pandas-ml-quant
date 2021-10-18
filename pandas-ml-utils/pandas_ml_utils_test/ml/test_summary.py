from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from pandas_ml_common.utils.column_lagging_utils import lag_columns
from pandas_ml_common_test.config import TEST_DF
from pandas_ml_utils import FittableModel, FeaturesLabels, SkModelProvider, FittingParameter
from pandas_ml_utils.ml.summary import ClassificationSummary, RegressionSummary, ReconstructionSummary, FeatureSelectionSummary


class TestSummary(TestCase):

    def test_classification(self):
        df = TEST_DF[-100:].copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(MLPClassifier(max_iter=100, random_state=42)),
                    FeaturesLabels(
                        features=["Close"],
                        features_postprocessor=lambda df: lag_columns(df.pct_change(), range(10)),
                        labels=["Close"],
                        labels_postprocessor=lambda df: df.pct_change().shift(-1) > 0,
                        label_type=int
                    ),
                    summary_provider=ClassificationSummary
                ),
                FittingParameter(epochs=1)
            )

        summary = fit.test_summary
        html = summary._repr_html_()

    def test_regression(self):
        df = TEST_DF[-100:].copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor(max_iter=100, random_state=42, activation='tanh')),
                    FeaturesLabels(
                        features=["Close"],
                        features_postprocessor=lambda df: lag_columns(df.pct_change(), range(10)),
                        labels=["Close"],
                        labels_postprocessor=lambda df: df.pct_change().shift(-1),
                        label_type=float
                    ),
                    summary_provider=RegressionSummary
                ),
                FittingParameter(epochs=1)
            )

        summary = fit.test_summary
        html = summary._repr_html_()

    def test_regression_reconstruction(self):
        df = TEST_DF[-100:].copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor(max_iter=100, random_state=42, activation='tanh')),
                    FeaturesLabels(
                        features=["Close"],
                        features_postprocessor=lambda df: lag_columns(df.pct_change(), range(10)),
                        labels=["Close"],
                        labels_postprocessor=lambda df: df.pct_change().shift(-1),
                        label_type=float
                    ),
                    summary_provider=ReconstructionSummary
                ),
                FittingParameter(epochs=1)
            )

        summary = fit.test_summary
        html = summary._repr_html_()

    def test_feature_selection(self):
        print(FeatureSelectionSummary)
        # raise NotImplemented
