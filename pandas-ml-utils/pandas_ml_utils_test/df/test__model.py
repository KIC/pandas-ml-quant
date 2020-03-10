from unittest import TestCase

from pandas_ml_utils.df.model import Model
from pandas_ml_common_test.config import TEST_DF


class TestDfModelExtension(TestCase):

    def test__property(self):
        self.assertIsInstance(TEST_DF.model, Model)

