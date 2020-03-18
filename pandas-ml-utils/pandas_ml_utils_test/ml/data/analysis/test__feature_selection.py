import unittest

from pandas_ml_utils import pd, FeaturesAndLabels


class TestFeatureSelection(unittest.TestCase):

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
