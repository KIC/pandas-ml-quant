from copy import deepcopy
from unittest import TestCase

from pandas_ml_common import LazyInit
from pandas_ml_common.utils.serialization_utils import serializeb, deserializeb


class TestLazyInit(TestCase):

    def test_copy(self):
        l = LazyInit(lambda: 12)

        self.assertIsNone(l.value)
        self.assertEqual(12, l())
        self.assertIsNotNone(l.value)

        l2 = deepcopy(l)
        self.assertIsNone(l2.value)
        self.assertEqual(12, l2())

    def test_serialization(self):
        l = LazyInit(lambda: 12)

        self.assertIsNone(l.value)
        self.assertEqual(12, l())
        self.assertIsNotNone(l.value)

        l2 = deserializeb(serializeb(l))
        self.assertIsNone(l2.value)
        self.assertEqual(12, l2())
