from unittest import TestCase

import numpy as np

from pandas_ml_common import pd, flatten_multi_column_index
from pandas_ml_common.utils import intersection_of_index, loc_if_not_none, add_multi_index, get_pandas_object, Constant, \
    same_columns_after_level, difference_in_index


class TestDfIndexUtils(TestCase):

    def test_get_pandas_obj(self):
        df = pd.DataFrame({"hallo": [1, 2, 3]})
        dfmi = df.copy()
        dfmi.columns = pd.MultiIndex.from_product([["a"], dfmi.columns])
        # print(dfmi)

        self.assertIsNone(get_pandas_object(df, None))
        self.assertListEqual([1, 2, 3], get_pandas_object(df, "hallo").to_list())
        self.assertListEqual([9, 9, 9], get_pandas_object(df, Constant(9)).to_list())
        self.assertListEqual([2., 4., 6.], get_pandas_object(df, 2.0, {float: lambda df, item, **kwargs: df["hallo"] * item}).to_list())
        self.assertListEqual([2, 4, 6], get_pandas_object(df, df["hallo"] * 2).to_list())
        self.assertListEqual([2, 4, 6], get_pandas_object(df, lambda df: df["hallo"] * 2).to_list())
        self.assertEqual((3, 0), get_pandas_object(df, "allo").shape)

        self.assertListEqual([1, 2, 3], get_pandas_object(dfmi, "hallo").iloc[:,0].to_list())
        self.assertListEqual([1, 2, 3], get_pandas_object(dfmi, "a").iloc[:,0].to_list())
        self.assertListEqual([1, 2, 3], get_pandas_object(dfmi, ".*ll.*").iloc[:, 0].to_list())
        self.assertEqual((3, 0), get_pandas_object(dfmi, "allo").shape)
        self.assertEqual((3, 1), get_pandas_object(dfmi, r"ha..o").shape)
        # print(get_pandas_object(dfmi, r"ha..o"))

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

    def test_add_multi_index(self):
        df = pd.DataFrame({}, index=[1, 2, 3, 4])
        df1 = add_multi_index(df, "A", axis=0)
        df2 = add_multi_index(df1, "B", axis=0, level=1)
        df3 = add_multi_index(df1, "B", axis=0, level=2)
        # print(df3)

        self.assertListEqual(df1.index.to_list(), [("A", 1), ("A", 2), ("A", 3), ("A", 4)])
        self.assertListEqual(df2.index.to_list(), [("A", "B", 1), ("A", "B", 2), ("A", "B", 3), ("A", "B", 4)])
        self.assertListEqual(df3.index.to_list(), [(1, "A", "B"), (2, "A", "B"), (3, "A", "B"), (4, "A", "B")])

    def test_similar_columns_multi_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4], columns=pd.MultiIndex.from_product([["a", "b"], range(3)]))
        df2 = pd.DataFrame({}, index=[1, 2, 3, 4], columns=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 3)]))

        self.assertTrue(same_columns_after_level(df1))
        self.assertFalse(same_columns_after_level(df2))

    def test_flatten_multiindex(self):
        df = pd.DataFrame({}, index=range(4), columns=pd.MultiIndex.from_product([["a", "b"], range(3)]))
        self.assertListEqual(
            [('a', 0), ('a', 1), ('a', 2), ('b', 0), ('b', 1), ('b', 2)],
            flatten_multi_column_index(df.copy()).columns.tolist()
        )
        self.assertListEqual(
            ['a, 0', 'a, 1', 'a, 2', 'b, 0', 'b, 1', 'b, 2'],
            flatten_multi_column_index(df.copy(), as_string=True).columns.tolist()
        )
        self.assertListEqual(
            [(12, 'a', 0), (12, 'a', 1), (12, 'a', 2), (12, 'b', 0), (12, 'b', 1), (12, 'b', 2)],
            flatten_multi_column_index(df.copy(), prefix=12).columns.tolist()
        )
        self.assertListEqual(
            ['12, a, 0', '12, a, 1', '12, a, 2', '12, b, 0', '12, b, 1', '12, b, 2'],
            flatten_multi_column_index(df.copy(), as_string=True, prefix=12).columns.tolist()
        )

    def test_difference(self):
        df1 = pd.DataFrame({"a": np.ones(3), "b": np.ones(3)})
        df2 = pd.DataFrame({"b": np.ones(3), "c": np.ones(3)})
        df3 = pd.DataFrame({"d": np.ones(3), "c": np.ones(3)})

        self.assertListEqual(
            [["a", "b"], ["c"], ["d"]],
            [i.to_list() for i in difference_in_index(df1, df2, df3, axis=1)]
        )

    def test_difference_with_series_index(self):
        df1 = pd.DataFrame({"a": np.ones(3), "b": np.ones(3)})
        s1 = pd.Series(np.ones(3), name="a")
        s2 = pd.Series(np.ones(3), name="b")
        df3 = pd.DataFrame({"d": np.ones(3), "c": np.ones(3)})

        self.assertListEqual(
            [[0, 1, 2], [], [], []],
            [i.to_list() for i in difference_in_index(df1, s1, s2, df3, axis=0)]
        )

    def test_difference_with_series_columns(self):
        df1 = pd.DataFrame({"a": np.ones(3), "b": np.ones(3)})
        s1 = pd.Series(np.ones(3), name="b")
        s2 = pd.Series(np.ones(3), name="c")
        df3 = pd.DataFrame({"d": np.ones(3), "c": np.ones(3)})

        self.assertListEqual(
            [['a', 'b'], [], ['c'], ['d']],
            [i.to_list() for i in difference_in_index(df1, s1, s2, df3, axis=1)]
        )