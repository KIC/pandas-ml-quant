from unittest import TestCase

from pandas import Timestamp

from pandas_ml_common import MLCompatibleValues
from pandas_ml_common_test.config import TEST_DF, TEST_MULTI_INDEX_DF, TEST_MUTLI_INDEX_ROW_DF


class TestML(TestCase):

    def test__property(self):
        self.assertIsInstance(TEST_DF.ML, MLCompatibleValues)
        self.assertIs(TEST_DF.to_frame(), TEST_DF)

    def test__llocproperty(self):
        self.assertEqual([('A', 'Close'), ('B', 'Close')], TEST_MULTI_INDEX_DF.ML.lloc["Close", 1].columns.to_list())
        self.assertEqual(
            [('A', Timestamp('2020-03-03 00:00:00')),
             ('A', Timestamp('2020-03-04 00:00:00')),
             ('A', Timestamp('2020-03-05 00:00:00')),
             ('B', Timestamp('2020-03-03 00:00:00')),
             ('B', Timestamp('2020-03-04 00:00:00')),
             ('B', Timestamp('2020-03-05 00:00:00'))
            ],
            TEST_MUTLI_INDEX_ROW_DF.ML.lloc["2020-03-03":"2020-03-05", :, 1].index.to_list()
        )
        self.assertEqual((6, 2), TEST_MUTLI_INDEX_ROW_DF.ML.lloc["2020-03-03":"2020-03-05", ["Close", "High"], 1].shape)
