import logging
from datetime import datetime

import cachetools
import investpy as ip
import numpy as np
import pandas as pd
from pangres import upsert

from pandas_ml_quant_data_provider.quant_data import DataProvider

_log = logging.getLogger(__file__)


class Investing(DataProvider):

    min_date = datetime(1970, 1, 2)

    def __init__(self):
        super().__init__(__file__)

    @cachetools.func.ttl_cache(maxsize=10000, ttl=10 * 60)
    def has_symbol(self, symbol: str, **kwargs):
        table_exists_query = f"SELECT name FROM sqlite_master WHERE name='{DataProvider.time_series_table_name}'"
        with self.engine.connect() as con:
            tables = con.execute(table_exists_query).first()

        if tables is None:
            self._build_index()

        symbols = list(con.execute(f"SELECT * from {DataProvider.time_series_table_name} WHERE symbol = '{symbol}'"))

        if len(symbols) < 0:
            _log.info(f"Symbol {symbol} not found! You might consider updating the index `Invsting().update_symbols()`")
            return False
        elif len(symbols) > 1:
            symbol_list = '\n'.join(symbols)
            _log.warning(f"Ambiguous {symbol} found more then once!\n{symbol_list}")
            return True
        else:
            return True

    @cachetools.func.ttl_cache(maxsize=100, ttl=10 * 60)
    def load(self, symbol: str, **kwargs):
        pass

    @cachetools.func.ttl_cache(maxsize=1, ttl=60 * 60 * 24)
    def update_symbols(self, **kwargs):
        self._build_index()

    def update_quotes(self, **kwargs):
        max_date = DataProvider.tomorrow().strftime('%d/%m/%Y')
        max_errors_in_a_row = kwargs['max_errors'] if 'max_errors' in kwargs else 25
        exception_count = 0

        with self.engine.connect() as con:
            if "resume" in kwargs and kwargs["resume"] is True:
                sql = f"""
                    SELECT * FROM {DataProvider.symbols_table_name} s
                     WHERE NOT EXISTS (SELECT * FROM {DataProvider.time_series_table_name} t 
                                        WHERE s.symbol = t.symbol
                                          AND s.type = t.type
                                          AND s.country = t.country LIMIT 1)
                """

                symbols = list(con.execute(sql))
                _log.info(f"resume download of {len(symbols)} assets")
            else:
                symbols = list(con.execute(f'SELECT * FROM {DataProvider.symbols_table_name}'))

        for asset in symbols:
            asset_type = asset["type"]

            # TODO could be
            #  con.execute(f"select date(max(date), '-2 day') form {DataProvider.time_series_table_name}
            #  where symbol = '{symbol}' and type = '{type}' and country = '{country}'").first() or Investing.min_date
            from_date = Investing.min_date.strftime('%d/%m/%Y')

            try:
                if asset_type == "BOND":
                    # name == symbol
                    df = ip.get_bond_historical_data(asset["symbol"], from_date=from_date, to_date=max_date)
                elif asset_type == "CERT":
                    df = ip.get_certificate_historical_data(asset['name'], country=asset["country"], from_date=from_date, to_date=max_date)
                elif asset_type == "CRYPTO":
                    df = ip.get_crypto_historical_data(asset['name'], from_date=from_date, to_date=max_date)
                elif asset_type == "COMM":
                    df = ip.get_commodity_historical_data(asset['symbol'], from_date=from_date, to_date=max_date)
                elif asset_type == "ETF":
                    df = ip.get_etf_historical_data(asset['name'], country=asset["country"], from_date=from_date, to_date=max_date)
                elif asset_type == "FUND":
                    df = ip.get_fund_historical_data(asset['name'], country=asset["country"], from_date=from_date, to_date=max_date)
                elif asset_type == "FX":
                    df = ip.get_currency_cross_historical_data(asset["symbol"], from_date=from_date, to_date=max_date)
                elif asset_type == "INDEX":
                    df = ip.get_index_historical_data(asset['name'], country=asset["country"], from_date=from_date, to_date=max_date)
                elif asset_type == "STOCK":
                    df = ip.get_stock_historical_data(asset['symbol'], country=asset["country"], from_date=from_date, to_date=max_date)

                # skip invalid downloads
                if df is None:
                    continue

                # fix columns
                if "Volume" not in df:
                    df["Volume"] = np.nan
                    df["Volume"] = df["Volume"].astype(float)

                if "Currency" not in df:
                    df["Currency"] = None
                    df["Currency"] = df["Currency"].astype(str)

                if "Exchange" not in df:
                    df["Exchange"] = None
                    df["Exchange"] = df["Exchange"].astype(str)

                # add asset information
                df.index = pd.MultiIndex.from_tuples([(
                    asset["symbol"],
                    asset_type,
                    asset["country"],
                    t
                ) for t in df.index]).rename(["symbol", "type", "country", "date"])

                # update price data in the database
                upsert(self.engine, df, DataProvider.time_series_table_name, if_row_exists='update')
                exception_count = 0

            except IndexError as ie:
                # ignore index error
                _log.error(f"unlised symbol: {asset} {ie}")
            except Exception as e:
                exception_count += 1
                _log.error(f"error for symbol: {asset} {e}")

                if exception_count >= max_errors_in_a_row:
                    raise ValueError("Something is really wrong!")

    def _build_index(self):
        def with_index(df, type, column, new_name="symbol"):
            if not "country" in df:
                df["country"] = "unknown"
            else:
                df["country"] = df["country"].replace({None: "unknown", "": "unknown"}).fillna('unknown')

            df.index = pd.MultiIndex.from_tuples(
                [(s, type, c) for s, c in zip(df[column].to_list(), df["country"].to_list())]
            ).rename([new_name if new_name is not None else column, "type", "country"])
            df = df.drop([column, "country"], axis=1)
            df = df.drop(df.index[df.index.duplicated('first')], axis=0)
            return df.loc[df.index.dropna()]

        symbols_df = pd.concat([
            with_index(ip.get_bonds(), "BOND", "name"),             # country	"name"	full_name
            with_index(ip.get_certificates(), "CERT", "symbol"),    # country', 'name', 'full_name', '"symbol"', 'issuer', 'isin', 'asset_class', 'underlying'
            with_index(ip.get_cryptos(), "CRYPTO", "symbol"),       # 'name', '"symbol"', 'currency'
            with_index(ip.get_commodities(), "COMM", "name"),       # 'title', 'country', '"name"', 'full_name', 'currency', 'group'
            with_index(ip.get_etfs(), "ETF", "symbol"),             # 'country', 'name', 'full_name', '"symbol"', 'isin', 'asset_class', 'currency', 'stock_exchange', 'def_stock_exchange'
            # with_index(ip.get_funds(), "FUND", "isin"),             # 'country', 'name', 'symbol', 'issuer', '"isin"', 'asset_class', 'currency', 'underlying'
            with_index(ip.get_indices(), "INDEX", "symbol"),        # 'country', 'name', 'full_name', '"symbol"', 'currency', 'class', 'market'
            with_index(ip.get_stocks(), "STOCK", "symbol"),         # ['country', 'name', 'full_name', 'isin', 'currency', '"symbol"'
            with_index(pd.DataFrame([f'{c}/USD' for c in ip.get_available_currencies()], columns=['symbol']), "FX", "symbol")
        ], axis=0)

        # update the index table
        upsert(self.engine, symbols_df, DataProvider.symbols_table_name, if_row_exists='ignore')


# for an initial data load we call it directly from here...
if __name__ == '__main__':
    # Investing().update_symbols()
    Investing().update_quotes(resume=True, max_errors=1500)
