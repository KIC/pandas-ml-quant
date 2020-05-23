import os
from unittest import TestCase

from pandas_ml_quant_data_provider import fetch_timeseries, YAHOO, INVESTING, CRYPTO_COMPARE, FRED, pd


class TestFetch(TestCase):

    def test__mixed(self):
        df = fetch_timeseries({
            YAHOO: ["SPY", "DIA"],
            INVESTING: ["index::NYSE Tick Index::united states", "bond::U.S. 30Y::united states"],
            CRYPTO_COMPARE: ["BTC"],
        })

        print(df.columns.to_list())
        print(df.head())
        print(len(df))

        self.assertListEqual(
            [('SPY', 'Open'), ('SPY', 'High'), ('SPY', 'Low'), ('SPY', 'Close'), ('SPY', 'Volume'), ('SPY', 'Dividends'), ('SPY', 'Stock Splits'),
             ('DIA', 'Open'), ('DIA', 'High'), ('DIA', 'Low'), ('DIA', 'Close'), ('DIA', 'Volume'), ('DIA', 'Dividends'), ('DIA', 'Stock Splits'),
             ('NYSE Tick Index', 'Open'), ('NYSE Tick Index', 'High'), ('NYSE Tick Index', 'Low'), ('NYSE Tick Index', 'Close'), ('NYSE Tick Index', 'Volume'), ('NYSE Tick Index', 'Currency'),
             ('U.S. 30Y', 'Open'), ('U.S. 30Y', 'High'), ('U.S. 30Y', 'Low'), ('U.S. 30Y', 'Close'),
             ('BTC', 'close'), ('BTC', 'high'), ('BTC', 'low'), ('BTC', 'open'), ('BTC', 'volumefrom'), ('BTC', 'volumeto')],
            df.columns.to_list()
        )

        self.assertGreaterEqual(len(df), 2400)

    def test__fred(self):

        if "FRED_API_KEY" not in os.environ:
            print("skipping FRED due to missing FRED_API_KEY key")
            return

        df = fetch_timeseries({FRED: ["GDP", "WALCL"]}, ffill=True)

        print(df.columns.to_list())
        print(df.head())
        print(len(df))

        self.assertListEqual(df.columns.to_list(), [('FRED', 'GDP'), ('FRED', 'WALCL')])
        self.assertGreaterEqual(len(df), 968)

    def test__cryptocompare(self):
        df = fetch_timeseries({"cryptocompare_hourly": "BTC"}, force_lower_case=True)
        print(df.tail())

        self.assertGreaterEqual(len(df), 1440)

    def test__monkey_patch(self):
        self.assertIsNotNone(pd.fetch_timeseries)
        self.assertIsNotNone(pd.fetch_yahoo)
        self.assertIsNotNone(pd.fetch_investing)
        self.assertIsNotNone(pd.fetch_cryptocompare_daily)

