from unittest import TestCase

from pandas_ml_common import MLCompatibleValues
from pandas_ml_common_test.config import TEST_DF


class TestML(TestCase):

    def test__property(self):
        self.assertIsInstance(TEST_DF.ML, MLCompatibleValues)
        self.assertIs(TEST_DF.to_frame(), TEST_DF)


