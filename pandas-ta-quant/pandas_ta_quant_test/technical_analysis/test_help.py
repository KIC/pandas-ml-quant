from unittest import TestCase

from pandas_ta_quant_test.config import DF_TEST


class TestHelp(TestCase):

    def test_help_frame(self):
        help = DF_TEST.ta.help

        self.assertGreater(len(help), 50)
        self.assertIn("doc", help)
