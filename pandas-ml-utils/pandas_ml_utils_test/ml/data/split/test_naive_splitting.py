from unittest import TestCase
import pandas as pd

from pandas_ml_utils.ml.data.splitting import NaiveSplitter


class TestNaiveSplitter(TestCase):

    def test_sampler(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        train_ix, test_ix = NaiveSplitter().train_test_split(df.index)

        """then"""
        print(train_ix, test_ix)
        self.assertListEqual([0,1,2], train_ix.tolist())
        self.assertListEqual([3,4], test_ix.tolist())

