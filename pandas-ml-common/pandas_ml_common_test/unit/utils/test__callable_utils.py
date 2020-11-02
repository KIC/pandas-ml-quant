import inspect
from functools import partial
from unittest import TestCase

from pandas_ml_common import pd, np
from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args, call_if_not_none


class TestCallableUtils(TestCase):

    def test_call_dynamic_args(self):
        """when"""
        arguments = list(range(4))
        l1 = lambda a: a
        l2 = lambda a, b: a + b
        l3 = lambda a, *b: a + sum(b)
        l4 = lambda *a: sum(a)
        def f1(a, b, c, d):
            return a + b + c + d
        def f2(a, *b, **kwargs):
            return a + sum(b)
        def f3(a, b):
            return a + b

        """then"""
        self.assertEqual(call_callable_dynamic_args(l1, *arguments), 0)
        self.assertEqual(call_callable_dynamic_args(l2, *arguments), 1)
        self.assertEqual(call_callable_dynamic_args(l3, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(l4, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f1, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f2, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f3, *arguments), 1)

    def test_call_dynamic_args_kwargs(self):
        """expect"""
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, 1, b=2))
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, a=1, b=2))
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, 1, 2))
        self.assertRaises(Exception, lambda: call_callable_dynamic_args(lambda a, b: True, 1))
        self.assertRaises(Exception, lambda: call_callable_dynamic_args(lambda a, b: True, 1, c=1))

    def test_call_if_not_none(self):
        df1 = pd.DataFrame({"a": [np.nan]})
        df2 = None

        self.assertIsNone(call_if_not_none(df2, 'dropna'))
        self.assertEqual(0, len(call_if_not_none(df1, 'dropna')))
        self.assertEqual(1, len(df1))

    def test_final_kwargs(self):
        kwargs = call_callable_dynamic_args(lambda x, y, **kwargs: list(kwargs.keys()), x=1, y=2, z=3, foo="bar")
        self.assertListEqual(['z', 'foo'], kwargs)
