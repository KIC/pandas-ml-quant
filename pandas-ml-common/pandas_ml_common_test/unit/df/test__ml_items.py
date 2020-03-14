from unittest import TestCase

from pandas_ml_common import pd, Constant

a_x, b_x, c_x = "a_x", "b_x", "c_x"
a_y, b_y, c_y = "a_y", "b_y", "c_y"



class TestMLValues(TestCase):

    def test__item_multi_index(self):
        df = pd.DataFrame({}, columns=pd.MultiIndex.from_product([[a_x, b_x, c_x], [a_y, b_y, c_y]]))

        # pandas_ml_common_test ordinary access single
        for col in df.columns.tolist():
            self.assertEqual(col, df.ml[col].name)

        # pandas_ml_common_test ordinary access multi
        cols = []
        for col in df.columns.tolist():
            cols.append(col)
            self.assertEqual(cols, df.ml[cols].columns.tolist())

        # pandas_ml_common_test 1st level
        self.assertListEqual([a_y, b_y, c_y], df.ml[a_x].columns.tolist())

        # pandas_ml_common_test 2nd level
        self.assertListEqual([(b_y, a_x), (b_y, b_x), (b_y, c_x)], df.ml[b_y].columns.tolist())

        # pandas_ml_common_test regex
        self.assertListEqual([(a_x, a_y), (a_x, b_y), (a_x, c_y), (b_x, a_y), (c_x, a_y)],
                             df.ml["a_."].columns.tolist())

    def test__item_normal_index(self):
        df = pd.DataFrame({}, columns=[a_x, b_y, c_y])

        # pandas_ml_common_test ordinary access
        for col in df.columns.tolist():
            self.assertEqual(col, df.ml[col].name)

        # pandas_ml_common_test ordinary access multi
        cols = []
        for col in df.columns.tolist():
            cols.append(col)
            self.assertEqual(cols, df.ml[cols].columns.tolist())

        # pandas_ml_common_test regex
        self.assertListEqual([a_x], df.ml["a_."].columns.tolist())
        self.assertListEqual([b_y, c_y], df.ml["..y"].columns.tolist())

    def test_mixins(self):
        df = pd.DataFrame({}, columns=[a_x, b_y, c_y])

        self.assertListEqual([a_x, b_y], df.ml[[a_x, df[b_y]]].columns.tolist())

    def test_callable(self):
        df = pd.DataFrame({}, columns=[a_x, b_y])

        self.assertListEqual([a_x, b_y], df.ml[lambda df: df[[a_x, b_y]]].columns.tolist())

    def test_callable_mixins(self):
        df = pd.DataFrame({}, columns=[a_x, a_y, b_y, c_y])

        self.assertListEqual([a_x, b_y, a_y], df.ml[[a_x, df[b_y], lambda df: df[a_y]]].columns.tolist())

    def test_constant(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        self.assertListEqual([12, 12, 12], df.ml[Constant(12)].values.tolist())