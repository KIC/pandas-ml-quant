from unittest import TestCase

from pandas_ml_common import pd
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.utils import intersection_of_index, loc_if_not_none, add_multi_index


class TestDfIndexUtils(TestCase):

    def test_intersection_of_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=[   2, 3, 4])
        df3 = pd.DataFrame({}, index=[1,    3, 4])

        index = intersection_of_index(df1, df2, df3)

        self.assertListEqual([3, 4], index.tolist())

    def test_loc_if_not_none(self):
        df1 = pd.DataFrame({"A": [1, 2, 3, 4]}, index=[1, 2, 3, 4])
        df2 = None

        self.assertEqual(1, loc_if_not_none(df1, 1).values[0])
        self.assertIsNone(loc_if_not_none(df2, 1))

    def test_intersection_of_tuples(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=[2, 3, 4])
        df3 = pd.DataFrame({}, index=[1, 3, 4])

        index1 = intersection_of_index(df1, MultiFrameDecorator([df2, df3], True))
        index2 = intersection_of_index(MultiFrameDecorator([df1, df2], True), df3)

        self.assertListEqual([3, 4], index1.tolist())
        self.assertListEqual([3, 4], index2.tolist())

    def test_add_multi_index(self):
        df = pd.DataFrame({}, index=[1, 2, 3, 4])
        df1 = add_multi_index(df, "A", axis=0)
        df2 = add_multi_index(df1, "B", axis=0, level=1)
        df3 = add_multi_index(df1, "B", axis=0, level=2)
        # print(df3)

        self.assertListEqual(df1.index.to_list(), [("A", 1), ("A", 2), ("A", 3), ("A", 4)])
        self.assertListEqual(df2.index.to_list(), [("A", "B", 1), ("A", "B", 2), ("A", "B", 3), ("A", "B", 4)])
        self.assertListEqual(df3.index.to_list(), [(1, "A", "B"), (2, "A", "B"), (3, "A", "B"), (4, "A", "B")])
