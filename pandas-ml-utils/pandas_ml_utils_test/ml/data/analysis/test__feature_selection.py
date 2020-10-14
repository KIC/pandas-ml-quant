import unittest

import numpy as np

from pandas_ml_common import random_splitter
from pandas_ml_utils import pd, FeaturesAndLabels
from pandas_ml_utils_test.config import DF_NOTES


class TestFeatureSelection(unittest.TestCase):

    def test_feature_selection_classification(self):
        df = DF_NOTES[730:790].copy()
        report = df.model.feature_selection(
            features_and_labels=FeaturesAndLabels(
                features=["variance", "skewness", "kurtosis", "entropy", lambda df: np.sqrt(df["skewness"]).rename("redundant")],
                labels=["authentic"],
                label_type=int
            ),
            training_data_splitter=random_splitter(0.5),
            rfecv_splits=2,
            forest_splits=2
        )
        
        print(report)
        # print(report.test_summary.validation_kpis.feature_importance)

    def test_feature_selection_regression(self):
        pass

if __name__ == '__main__':
    unittest.main()
