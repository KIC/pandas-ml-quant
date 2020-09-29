import unittest
from unittest import TestCase

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from pandas_ml_common import naive_splitter, random_splitter
from pandas_ml_utils import FeaturesAndLabels, SkModel
from pandas_ml_utils.constants import *
from pandas_ml_utils_test.config import DF_NOTES


class TestModel(TestCase):

    def test_simple_classification_model(self):
        df = DF_NOTES.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 12), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=["variance", "skewness", "kurtosis", "entropy"],
                    labels=["authentic"],
                    label_type=bool
                )
            ),
            naive_splitter()
        )

        print(fit)
        html = fit._repr_html_()

        prediction = df.model.predict(fit.model)
        print(prediction)
        self.assertGreaterEqual(prediction[PREDICTION_COLUMN_NAME].iloc[-1].values, 0.68)

        backtest = df.model.backtest(fit.model)
        self.assertIn(FEATURE_COLUMN_NAME, backtest.df)
        self.assertIn(LABEL_COLUMN_NAME, backtest.df)
        np.testing.assert_array_almost_equal(prediction[PREDICTION_COLUMN_NAME].iloc[-1].values,
                                             backtest.df[PREDICTION_COLUMN_NAME].iloc[-1].values)

        # test multiple samples
        samples = df.model.predict(fit.model, samples=2)
        self.assertIsInstance(samples[PREDICTION_COLUMN_NAME].iloc[-1, 0], list)
        self.assertEqual(2, len(samples[PREDICTION_COLUMN_NAME].iloc[-1, 0]))

    @unittest.skip("cross validation needs to be re-implemented")
    def test_simple_classification_cross_validation(self):
        df = DF_NOTES.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 12), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=["variance", "skewness", "kurtosis", "entropy"],
                    labels=["authentic"],
                    label_type=bool
                )
            ),
            training_data_splitter=random_splitter(),
            cross_validation=KFold(3, random_state=42)
        )

        print(fit)
        html = fit._repr_html_()

        prediction = df.model.predict(fit.model)
        print(prediction)
        self.assertGreaterEqual(prediction[PREDICTION_COLUMN_NAME].iloc[-1].values, 0.7)

    def test_simple_classification_model_with_all_options(self):
        df = DF_NOTES.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(20, 12), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=["variance", "skewness", "kurtosis", "entropy"],
                    sample_weights=["variance"],
                    gross_loss=["kurtosis"],
                    targets=["entropy"],
                    labels=["authentic"],
                    label_type=bool
                )
            ),
            naive_splitter()
        )

        # should not thro an error
        html = fit._repr_html_()

        # fit resulting columns
        print(fit.test_summary.df)
        self.assertEqual(int(len(df) * 0.7), len(fit.training_summary.df))
        self.assertEqual(len(df) - int(len(df) * 0.7), len(fit.test_summary.df))
        # FIXME expect gross_loss and weights as well
        self.assertIn(FEATURE_COLUMN_NAME, fit.training_summary.df)
        self.assertIn(LABEL_COLUMN_NAME, fit.training_summary.df)
        self.assertIn(TARGET_COLUMN_NAME, fit.training_summary.df)

        self.assertIn(FEATURE_COLUMN_NAME, fit.test_summary.df)
        self.assertIn(LABEL_COLUMN_NAME, fit.test_summary.df)
        self.assertIn(TARGET_COLUMN_NAME, fit.test_summary.df)

        # prediction resulting columns
        prediction = df.model.predict(fit.model)
        print(prediction)
        self.assertIn(FEATURE_COLUMN_NAME, prediction)
        self.assertIn(TARGET_COLUMN_NAME, prediction)

        # backtest resulting columns
        # FIXME expect gross_loss and weights as well
        backtest = df.model.backtest(fit.model)
        print(backtest.df)
        self.assertEqual(len(df), len(backtest.df))
        self.assertIn(FEATURE_COLUMN_NAME, backtest.df)
        self.assertIn(LABEL_COLUMN_NAME, backtest.df)
        self.assertIn(TARGET_COLUMN_NAME, backtest.df)