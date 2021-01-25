from typing import Callable
from unittest import TestCase
from pandas_ta_quant import pd, np


class TestPatching(TestCase):

    def test_patched_pandas(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        self.assertIsInstance(df.ta.sma, Callable)

    def test_help_frame(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        help = df.ta.help

        self.assertGreater(len(help), 50)
        self.assertIn("doc", help)
