from unittest import TestCase

from pandas_ml_utils.df_patching.model_patch import DfModelPatch
from pandas_ml_utils_test.config import DF_TEST


class TestDfModelExtension(TestCase):

    def test__property(self):
        self.assertIsInstance(DF_TEST.model, DfModelPatch)

