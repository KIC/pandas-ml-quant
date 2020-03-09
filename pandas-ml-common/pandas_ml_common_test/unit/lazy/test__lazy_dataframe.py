import unittest
from copy import deepcopy

from pandas_ml_common import LazyDataFrame, pd


class TestLazyDataFrame(unittest.TestCase):

    def test_lazy_dataframe(self):
        df = pd.DataFrame({"pandas_ml_common_test": [1,2,3,4,5]})
        df2 = pd.DataFrame({"test2": [1,2,3,4,5]})
        ldf = LazyDataFrame(df, foo=lambda _df: _df[_df.columns[0]] * 2)
        ldf2 = ldf.with_dataframe(df2)

        # pandas_ml_common_test lazy functions
        pd.testing.assert_series_equal(ldf["foo"], df["pandas_ml_common_test"] * 2, check_names=False)
        self.assertEqual(ldf[["foo", "pandas_ml_common_test"]].shape, (5,2))
        pd.testing.assert_frame_equal(ldf[["pandas_ml_common_test", "foo"]], ldf.to_dataframe())
        pd.testing.assert_frame_equal(ldf2[["test2", "foo"]], ldf2.to_dataframe())
        pd.testing.assert_frame_equal(ldf[["foo"]], ldf2[["foo"]])

        # pandas_ml_common_test delegation
        pd.testing.assert_index_equal(ldf.index, df.index)
        self.assertEqual(len(ldf.min().index), 2)

        # pandas_ml_common_test structure
        self.assertFalse("foo" in ldf.df)
        self.assertTrue("foo" in ldf)

    def test_deepcopy(self):
        """given"""
        ldf = LazyDataFrame(None, foo=lambda _df: _df["lala"] * 2)

        """when"""
        copy = deepcopy(ldf)

        """then"""
        self.assertListEqual(list(copy.kwargs.keys()), list(ldf.kwargs.keys()))
        self.assertNotEqual(copy.hash, ldf.hash)


if __name__ == '__main__':
    unittest.main()
