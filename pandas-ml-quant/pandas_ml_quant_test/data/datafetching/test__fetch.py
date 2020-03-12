import logging
import unittest

from pandas_ml_quant import pd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ComponentTest(unittest.TestCase):

    def test_fetch_yahoo(self):
        """when"""
        df = pd.fetch_yahoo(spy="SPY").tail()
        print(df.columns)

        """then"""
        self.assertTrue(df["spy_Close"].sum() > 0)


    def test_crypto_compare(self):
        """when"""
        df = pd.fetch_cryptocompare_daily(coin="BTC", limit=None).tail()
        print(df.columns)

        """then"""
        self.assertTrue(df["close"].sum() > 0)

