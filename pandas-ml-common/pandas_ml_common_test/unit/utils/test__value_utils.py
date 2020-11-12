from unittest import TestCase

from pandas_ml_common.utils import to_pandas, unpack_nested_arrays, wrap_row_level_as_nested_array, hexplode
import pandas as pd
import numpy as np


class TestValueUtils(TestCase):

    def test_to_pandas_single(self):
        """given a numpy array"""
        arr = np.array([1,1,1,1,1])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1])

        """then the frame looks like this"""
        self.assertEqual(1, len(df.columns))
        self.assertListEqual([1,1,1,1,1], df[1].values.tolist())

    def test_to_pandas_simple_2d(self):
        """given a numpy array"""
        arr = np.array([
            [1],
            [1],
            [1],
            [1],
            [1],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1])

        """then the frame looks like this"""
        self.assertEqual(1, len(df.columns))
        self.assertListEqual([1,1,1,1,1], df[1].values.tolist())

    def test_to_pandas_simple_2d_2samples(self):
        """given a numpy array"""
        arr = np.array([
            [[1], [2]],
            [[1], [2]],
            [[1], [2]],
            [[1], [2]],
            [[1], [2]],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1])

        """then the frame looks like this"""
        self.assertEqual(1, len(df.columns))
        self.assertListEqual([[1, 2],[1, 2],[1, 2],[1, 2],[1, 2]], df[1].values.tolist())

    def test_to_pandas_embedded(self):
        """given a numpy array"""
        arr = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1, 2])

        """then the frame looks like this"""
        self.assertEqual(2, len(df.columns))
        self.assertListEqual([1,1,1,1,1], df[1].values.tolist())
        self.assertListEqual([2, 3], df[2].iloc[-1])

    def test_to_pandas_3d_single(self):
        """given a numpy array"""
        arr = np.array([
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1])

        """then the frame looks like this"""
        print(df)
        self.assertEqual(1, len(df.columns))
        self.assertListEqual([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]], df[1].values.tolist())

    def test_to_pandas_3d(self):
        """given a numpy array"""
        arr = np.array([
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1,2,3])

        """then the frame looks like this"""
        print(df)
        self.assertEqual(3, len(df.columns))
        self.assertListEqual([1,1,1,1,1], df[1].values.tolist())
        self.assertListEqual([2,2,2,2,2], df[2].values.tolist())
        self.assertListEqual([3,3,3,3,3], df[3].values.tolist())

    def test_to_pandas_3d_2samples(self):
        """given a numpy array"""
        arr = np.array([
            [[1, 2, 3], [3, 2, 1]],
            [[1, 2, 3], [3, 2, 1]],
            [[1, 2, 3], [3, 2, 1]],
            [[1, 2, 3], [3, 2, 1]],
            [[1, 2, 3], [3, 2, 1]],
        ])

        """when converted to a pandas data frame"""
        df = to_pandas(arr, index=[1,2,3,4,5], columns=[1,2,3])

        """then the frame looks like this"""
        self.assertEqual(3, len(df.columns))
        self.assertListEqual([[1, 3],[1, 3],[1, 3],[1, 3],[1, 3]], df[1].values.tolist())
        self.assertListEqual([[2,2],[2,2],[2,2],[2,2],[2,2]], df[2].values.tolist())
        self.assertListEqual([[3,1],[3,1],[3,1],[3,1],[3,1]], df[3].values.tolist())

    def test_nested_values(self):
        """given a symetrical nested array"""
        df = pd.DataFrame({
            "a": [[1, 2] for _ in range(5)],
            "b": [[1, 2] for _ in range(5)],
        })

        """when extracted then shape is 5,2,2"""
        self.assertEqual((5, 2, 2), unpack_nested_arrays(df).shape)

    def test_nested_values_invalid_shape(self):
        """given a non-symetrical nested array"""
        df = pd.DataFrame({
            "a": [[1, 2] for _ in range(5)],
            "b": [[1, 2, 3] for _ in range(5)],
        })

        """when extracted then shape can not be derived"""
        self.assertRaises(ValueError, lambda: unpack_nested_arrays(df))

    def test_nested_values_column_multiindex(self):
        """given a symetrical nested array"""
        df = pd.DataFrame(pd.DataFrame([
            [np.array([1, 2]) for _ in range(5)],
            [np.array([1, 2]) for _ in range(5)],
            [np.array([1, 2]) for _ in range(5)],
            [np.array([1, 2]) for _ in range(5)]
        ]).T.values, columns=pd.MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0), ("B", 1)]))

        print(df)
        """when extracted then shape is 5,2,2"""
        self.assertEqual((5, 4, 2), unpack_nested_arrays(df).shape)

    def test_nested_values_row_multiindex(self):
        """given a row-MultiIndex DataFrame"""
        df = pd.DataFrame(
            np.ones((10, 3)),
            index=pd.MultiIndex.from_tuples([
                *[("A", i) for i in range(7)],
                *[("B", i) for i in range(7, 10)],
            ])
        )

        """when extracting values"""
        values = unpack_nested_arrays(df)

        """then we have a list of numpy arrays"""
        self.assertEqual(2, len(values))
        self.assertEqual((7, 3), values[0].shape)
        self.assertEqual((3, 3), values[1].shape)

    def test_nesting_arrays(self):
        df = pd.DataFrame(
            np.random.random((10, 2)),
            columns=["a", "b"],
            index=pd.MultiIndex.from_product([["A", "B"], range(5)])
        )

        wrapped = wrap_row_level_as_nested_array(df, -1)
        self.assertEqual((2, 1), wrapped.shape)
        self.assertNotIsInstance(wrapped, pd.MultiIndex)

    def test_hexplode(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["b"] = [[1, 1], [2, 2], [3, 3]]

        res = hexplode(df, "b", ["b1", "b2"])
        self.assertEqual((3, 3), res.shape)
        self.assertEqual(9, res.iloc[-1].sum().item())