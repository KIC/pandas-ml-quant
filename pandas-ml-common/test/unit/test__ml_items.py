from unittest import TestCase

from pandas_ml_common import pd

a_x, b_x, c_x = "a_x", "b_x", "c_x"
a_y, b_y, c_y = "a_y", "b_y", "c_y"



class TestMLValues(TestCase):

    def test__item_multi_index(self):
        df = pd.DataFrame({}, columns=pd.MultiIndex.from_product([[a_x, b_x, c_x], [a_y, b_y, c_y]]))

        # test ordinary access single
        for col in df.columns.tolist():
            self.assertEqual(col, df.ml[col].name)

        # test ordinary access multi
        cols = []
        for col in df.columns.tolist():
            cols.append(col)
            self.assertEqual(cols, df.ml[cols].columns.tolist())

        # test 1st level
        self.assertListEqual([a_y, b_y, c_y], df.ml[a_x].columns.tolist())

        # test 2nd level
        self.assertListEqual([(a_x, b_y), (b_x, b_y), (c_x, b_y)], df.ml[b_y].columns.tolist())

        # test regex
        self.assertListEqual([(a_x, a_y), (a_x, a_y), (a_x, b_y), (a_x, c_y), (b_x, a_y), (c_x, a_y)],
                             df.ml["a_."].columns.tolist())

    def test__item_normal_index(self):
        df = pd.DataFrame({}, columns=[a_x, b_y, c_y])

        # test ordinary access
        for col in df.columns.tolist():
            self.assertEqual(col, df.ml[col].name)

        # test ordinary access multi
        cols = []
        for col in df.columns.tolist():
            cols.append(col)
            self.assertEqual(cols, df.ml[cols].columns.tolist())

        # test regex
        self.assertListEqual([a_x], df.ml["a_."].columns.tolist())
        self.assertListEqual([b_y, c_y], df.ml["..y"].columns.tolist())

    def test_mixins(self):
        df = pd.DataFrame({}, columns=[a_x, b_y, c_y])

        self.assertListEqual([a_x, b_y], df.ml[[a_x, df[b_y]]].columns.tolist())
