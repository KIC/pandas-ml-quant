from unittest import TestCase

from pandas_ml_common import pd, np
from pandas_ml_common.sampling import naive_splitter, random_splitter, dummy_splitter
from pandas_ml_common.sampling.splitter import stratified_random_splitter


class TestTrainTestData(TestCase):

    def test_naive_splitter(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        train_ix, test_ix = naive_splitter(0.3)(df.index)

        """then"""
        print(train_ix, test_ix)
        self.assertListEqual([0, 1, 2], train_ix.tolist())
        self.assertListEqual([3, 4], test_ix.tolist())

    def test_no_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "labelA": [1,2,3,4,5]})

        """when"""
        train_ix, test_ix = random_splitter(0)(df.index)
        train_ix2, test_ix2 = dummy_splitter(df.index)

        """then"""
        np.testing.assert_array_almost_equal(train_ix.values, df.index.values)
        np.testing.assert_array_almost_equal(train_ix.values, train_ix2.values)
        self.assertEqual(0, len(test_ix))

    def test_make_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        train_ix, test_ix = random_splitter(test_size=0.5)(df.index)

        """then"""
        self.assertEqual(2, len(train_ix))
        self.assertEqual(3, len(test_ix))

    def test_youngest_portion(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        train_ix, test_ix = random_splitter(test_size=0.6, youngest_size=0.25)(df.index)

        "then"
        self.assertEqual(6, len(test_ix))
        np.testing.assert_array_equal(test_ix[-2:], np.array([8, 9]))

    def test_stratified_random_splitter(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3]})

        """when"""
        train_ix, test_ix = stratified_random_splitter(test_size=0.5)(df.index, None, df[["labelA"]])

        """then each class is represented similarly often in each train and test set"""
        self.assertIn(2, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(2, df.loc[test_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[test_ix, "labelA"].to_list())
