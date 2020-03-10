from unittest import TestCase

from pandas_ml_common import pd
from pandas_ml_common.utils import intersection_of_index, loc_if_not_none


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

