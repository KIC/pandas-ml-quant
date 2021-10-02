from unittest import TestCase

import numpy as np
from sklearn.metrics import confusion_matrix

from pandas_ml_utils import FeaturesLabels, LambdaModel, FittableModel
from pandas_ml_utils.constants import *
from pandas_ml_utils_test.config import DF_NOTES


class TestLambdaModel(TestCase):

    def test_lambda_model_classifier(self):
        df = DF_NOTES.copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    LambdaModel(lambda f: np.abs(f[0].values) > 3.5),
                    FeaturesLabels(features=["variance"], labels=["authentic"])
                )
            )

        cm = confusion_matrix(
            fit.test_summary.df[LABEL_COLUMN_NAME].values,
            fit.test_summary.df[PREDICTION_COLUMN_NAME].values
        )

        tn, fp, fn, tp = cm.ravel()
        self.assertEqual(89, tp)
