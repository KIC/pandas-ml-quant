from unittest import TestCase

from pandas_ml_utils import pd, np
from pandas_ml_utils.ml.data.splitting import train_test_split


class TestTrainTestData(TestCase):

    def test_no_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "labelA": [1,2,3,4,5]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, 0)

        """then"""
        np.testing.assert_array_almost_equal(train_ix.values, df.index.values)
        self.assertEqual(0, len(test_ix))

    def test_make_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, test_size=0.5)

        """then"""
        self.assertEqual(2, len(train_ix))
        self.assertEqual(3, len(test_ix))

    def test_youngest_portion(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, test_size=0.6, youngest_size=0.25)

        "then"
        self.assertEqual(6, len(test_ix))
        np.testing.assert_array_equal(test_ix[-2:], np.array([8, 9]))
