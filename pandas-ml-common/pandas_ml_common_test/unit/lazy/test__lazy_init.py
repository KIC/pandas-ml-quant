import os
from unittest import TestCase

from keras import backend as K

from pandas_ml_common import LazyInit
from pandas_ml_common.serialization_utils import serialize, deserialize

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestLazyInit(TestCase):

    def test_serialization(self):
        """given"""
        lazy_val = LazyInit(lambda: K.constant(12))
        val = lazy_val()

        """when"""
        serialize(lazy_val, '/tmp/pandas_ml_common_test.dill')
        lazy_val_2 = deserialize('/tmp/pandas_ml_common_test.dill', LazyInit)

        """then"""
        self.assertEqual(12, K.eval(val))
        self.assertEqual(12, K.eval(lazy_val_2()))
