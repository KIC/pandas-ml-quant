from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_common.utils.numpy_utils import one_hot
from pandas_ml_quant.model.summary import WeightedClassificationSummary
from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
from pandas_ml_quant_test.config import DF_TEST_MULTI_CLASS, DF_TEST_MULTI
from pandas_ml_utils.constants import *


class TestSummary(TestCase):

    def test_multi_class_summary_odd_targets(self):
        # df = DF_TEST_MULTI_CASS
        target = np.arange(1, 4)
        price = np.arange(0, 4) + 0.5
        labels = [one_hot(np.random.randint(0, len(target)), len(target) + 1).tolist() for _ in range(len(price))]

        predictions = range(len(price))
        expected_losses = [0.0, -0.5, -0.5, 0]
        for prediction, expected_loss in zip(predictions, expected_losses):
            df = pd.DataFrame([target for _ in range(len(price))])
            df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns])
            df[(GROSS_LOSS_COLUMN_NAME, "loss")] = price
            df[(PREDICTION_COLUMN_NAME, "prediction")] = [one_hot(prediction, len(target) + 1).tolist() for _ in range(len(price))]
            df[(LABEL_COLUMN_NAME, "label")] = labels

            s = WeightedClassificationSummary(df, None, classes=len(target) + 1)
            self.assertEqual(s.df_gross_loss["loss"].clip(upper=0).sum(), expected_loss)
            print(s._gross_confusion())

    def test_multi_class_summary_even_targets(self):
        # df = DF_TEST_MULTI_CASS
        target = np.arange(1, 5)
        price = np.arange(0, 5) + 0.5
        labels = [one_hot(np.random.randint(0, len(target)), len(target) + 1).tolist() for _ in range(len(price))]

        predictions = range(len(price))
        expected_losses = [0.0, -0.5, -0.5, 0]
        for prediction, expected_loss in zip(predictions, expected_losses):
            df = pd.DataFrame([target for _ in range(len(price))])
            df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns])
            df[(GROSS_LOSS_COLUMN_NAME, "loss")] = price
            df[(PREDICTION_COLUMN_NAME, "prediction")] = [one_hot(prediction, len(target) + 1).tolist() for _ in range(len(price))]
            df[(LABEL_COLUMN_NAME, "label")] = labels
            print(df)

            s = WeightedClassificationSummary(df, None, classes=len(target) + 1)
            print(s.df_gross_loss)

    def test_multi_class_summary(self):
        df = DF_TEST_MULTI_CLASS
        s = WeightedClassificationSummary(df, None)
        print(df[TARGET_COLUMN_NAME].shape)
        print(s._gross_confusion())

