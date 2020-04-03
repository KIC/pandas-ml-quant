from unittest import TestCase

from pandas_ml_common.utils import to_pandas
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
