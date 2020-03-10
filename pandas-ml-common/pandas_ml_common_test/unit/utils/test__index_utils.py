from unittest import TestCase

from pandas_ml_common import pd
from pandas_ml_common.utils import intersection_of_index


class TestDfIndexUtils(TestCase):

    def test_intersection_of_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=[   2, 3, 4])
        df3 = pd.DataFrame({}, index=[1,    3, 4])

        index = intersection_of_index(df1, df2, df3)

        self.assertListEqual([3, 4], index.tolist())

