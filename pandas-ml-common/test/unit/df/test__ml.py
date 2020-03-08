from unittest import TestCase

from pandas_ml_common import ML
from test.config import TEST_DF


class TestML(TestCase):

    def test__property(self):
        self.assertIsInstance(TEST_DF.ml, ML)
