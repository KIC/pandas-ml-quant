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

    def test_naive_splitter_multi_index_row(self):
        """given"""
        df = pd.DataFrame({"featureA": range(10),
                           "labelA": range(10)})

        df.index = pd.MultiIndex.from_product([["A", "B"], range(5)])

        """when"""
        train_ix, test_ix = naive_splitter(0.3, partition_row_multi_index=True)(df.index)

        """then"""
        print(train_ix.tolist(), test_ix.tolist())
        self.assertListEqual([('A', 0), ('A', 1), ('A', 2), ('B', 0), ('B', 1), ('B', 2)], train_ix.tolist())
        self.assertListEqual([('A', 3), ('A', 4), ('B', 3), ('B', 4)], test_ix.tolist())

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

    def test_random_splitter_multi_index_row(self):
        """given"""
        df = pd.DataFrame({"featureA": range(10),
                           "labelA": range(10)})

        df.index = pd.MultiIndex.from_product([["A", "B"], range(5)])

        """when"""
        train_ix, test_ix = random_splitter(test_size=0.6, youngest_size=0.25, partition_row_multi_index=True)(df.index)
        print(train_ix.tolist(), test_ix.tolist())

        """then"""
        self.assertEqual(8, len(test_ix))
        self.assertIn(('A', 4), test_ix)
        self.assertIn(('B', 4), test_ix)

    def test_stratified_random_splitter(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3]})

        """when"""
        train_ix, test_ix = stratified_random_splitter(test_size=0.5)(df.index, y=df[["labelA"]])

        """then each class is represented similarly often in each train and test set"""
        self.assertIn(2, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(2, df.loc[test_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[test_ix, "labelA"].to_list())

    def test_stratified_random_splitter_multi_index_row(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                           "labelA": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3] * 2})

        df.index = pd.MultiIndex.from_product([["A", "B"], range(10)])

        """when"""
        train_ix, test_ix = stratified_random_splitter(test_size=0.5, partition_row_multi_index=True)(df.index, y=df[["labelA"]])
        print(train_ix.tolist(), test_ix.tolist())

        """then each class is represented similarly often in each train and test set"""
        self.assertIn(2, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[train_ix, "labelA"].to_list())
        self.assertIn(2, df.loc[test_ix, "labelA"].to_list())
        self.assertIn(3, df.loc[test_ix, "labelA"].to_list())
