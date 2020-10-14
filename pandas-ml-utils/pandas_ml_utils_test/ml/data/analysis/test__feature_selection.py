import unittest

import numpy as np

from pandas_ml_common import random_splitter
from pandas_ml_utils import pd, FeaturesAndLabels
from pandas_ml_utils_test.config import DF_NOTES


class TestFeatureSelection(unittest.TestCase):

    def test_feature_selection_clasification(self):
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
        print(report.test_summary.validation_kpis.feature_importance)

    def test_feature_selection(self):
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "featureB": [5, 4, 3, 2, 1],
                           "featureC": [1, 2, 1, 2, 1],
                           "labelA": [1, 2, 3, 4, 5],
                           "labelB": [5, 4, 3, 2, 1]})


        analysis = df.model.feature_selection(FeaturesAndLabels(["featureA", "featureB", "featureC"], ["labelA"]),
                                              lags=[2], show_plots=False)


        print(analysis)
        # top features are A, B, C
        self.assertListEqual(["featureA", "featureB", "featureC"], analysis[0])
        self.assertListEqual([0, 1], analysis[1])


if __name__ == '__main__':
    unittest.main()
