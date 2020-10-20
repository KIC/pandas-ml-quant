from unittest import TestCase

import numpy as np

from pandas_ml_common import MLCompatibleValues, pd
from pandas_ml_common_test.config import TEST_DF, TEST_MULTI_INDEX_DF


class TestMLValues(TestCase):

    def test__property(self):
        self.assertIsInstance(TEST_DF._, MLCompatibleValues)

    def test__values(self):
        df = TEST_DF[-5:].copy()
        df["multi_1"] = df["Close"].apply(lambda v: [v, v])
        df["deep_1"] = [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))]

        self.assertEqual((5, 2), df["multi_1"]._.values.shape)
        self.assertEqual((5, 3, 2), df[["multi_1", "multi_1", "multi_1"]]._.values.shape)

        self.assertEqual((5, 2, 4), df["deep_1"]._.values.shape)
        self.assertEqual((5, 1, 2, 4), df[["deep_1"]]._.values.shape)
        self.assertEqual((5, 3, 2, 4), df[["deep_1", "deep_1", "deep_1"]]._.values.shape)

        self.assertRaises(Exception, lambda: df[["deep_1", "multi_1"]]._.values.shape)

    def test__values_multi_index(self):
        df = TEST_MULTI_INDEX_DF[-5:].copy()

        self.assertEqual((5, 2, 6), df._.values.shape)
        self.assertEqual(df._.values[0, 0, -1], df._.values[0, 1, -1])
        self.assertEqual(238703600, df._.values[0, 1, -1])

    def test_multi_index_nested_values(self):
        df = pd.DataFrame(
            {
                ("A", "a"): [1, 2, 3, 4, 5],
                ("A", "b"): [3, 2, 1, 0, 0],
                ("A", "c"): [3, 2, 1, 0, 0],
                ("B", "a"): [1, 2, 3, 1, 2],
                ("B", "b"): [3, 2, 1, 0, 1],
                ("B", "c"): [3, 2, 1, 0, 1],
                ("C", "a"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("C", "b"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("C", "c"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("D", "a"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
            },
            index=[1, 2, 3, 4, 5],
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns.tolist())

        """when"""
        print(df)
        rnnShape = df[["A"]]._.values
        rnnShape2 = df[["A", "B"]]._.values
        rnnShapeExt = df["C"]._.values
        labelShape = df["D"]._.values

        """then"""
        print(rnnShape.shape, rnnShape2.shape, rnnShapeExt.shape, labelShape.shape)
        self.assertEqual((5, 1, 3), rnnShape.shape)
        self.assertEqual((5, 2, 3), rnnShape2.shape)
        self.assertEqual((5, 3, 2, 4), rnnShapeExt.shape)
        self.assertEqual((5, 1, 2, 4), labelShape.shape)

