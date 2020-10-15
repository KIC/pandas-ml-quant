import unittest

from sklearn.datasets import make_classification

from pandas_ml_common import stratified_random_splitter
from pandas_ml_utils import pd, FeaturesAndLabels


class TestFeatureSelection(unittest.TestCase):

    def test_feature_selection_classification(self):
        data = make_classification(n_samples=20, n_features=5, n_informative=4, n_redundant=1, n_classes=2)
        df = pd.DataFrame(data[0])
        df["label"] = data[1]

        report = df.model.feature_selection(
            features_and_labels=FeaturesAndLabels(
                features=list(range(5)),
                labels=["label"],
                label_type=int
            ),
            training_data_splitter=stratified_random_splitter(0.5),
            rfecv_splits=2,
            forest_splits=2
        )
        
        print(report)
        # print(report.test_summary.validation_kpis.feature_importance)

    def test_feature_selection_regression(self):
        pass


if __name__ == '__main__':
    unittest.main()
