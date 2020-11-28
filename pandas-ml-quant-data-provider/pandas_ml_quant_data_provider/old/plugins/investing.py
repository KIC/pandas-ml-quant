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
    prefered_countries = ['united states', 'canada', 'euro zone', 'united kingdom']
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

        with self.engine.connect() as con:
            assets = list(con.execute(f"SELECT * from {DataProvider.symbols_table_name} WHERE symbol = '{symbol}'"))

        if len(assets) < 0:
            _log.info(f"Symbol {symbol} not found! You might consider updating the index `Invsting().update_symbols()`")
            return False
        elif len(assets) > 1:
            symbol_list = '\n'.join([str(a) for a in assets])
            _log.warning(f"Ambiguous {symbol} found more then once!\n{symbol_list}")
            return True
        else:
            return True

    def load(self, symbol: str, **kwargs):
        # select symbols, rank countries, raise error on multiple type
        symbol_detail = symbol.split("/")
        if len(symbol_detail) > 2:
            symbol, type, country = symbol_detail
            with self.engine.connect() as con:
                assets = list(con.execute(f"SELECT * FROM {DataProvider.symbols_table_name} "
                                          f" WHERE symbol = '{symbol}'"
                                          f"   AND type = '{type}'"
                                          f"   AND country = '{country}'"))

        elif len(symbol_detail) > 1:
            symbol, country = symbol_detail
            with self.engine.connect() as con:
                assets = list(con.execute(f"SELECT * FROM {DataProvider.symbols_table_name} "
                                          f" WHERE symbol = '{symbol}'"
                                          f"   AND country = '{country}'"))
        else:
            with self.engine.connect() as con:
                assets = list(con.execute(f"SELECT * FROM {DataProvider.symbols_table_name}"
                                          f" WHERE symbol = '{symbol}'"))

        if len(assets) < 0:
            return None
        elif len(assets) > 1:
            # ambiguity check
            types = {a["type"] for a in assets}
            if len(types) != 1:
                raise ValueError(f"Invalid or ambiguous symbol: {symbol} types: {types}")

            by_countries = {a["country"]: a for a in assets}
            if len(by_countries) > 1:
                preferred_countries = [by_countries[c] for c in Investing.prefered_countries if c in by_countries]
                if len(preferred_countries) < 1 < len(by_countries):
                    raise ValueError(f"ambiguous country: {by_countries.keys()}")

                asset = preferred_countries[0]
                _log.warning(f"ambiguous country: {by_countries.keys()}\ndefault to {asset['country']}")
        else:
            asset = assets[0]

        # date range
        from_date = Investing.min_date.strftime('%d/%m/%Y')
        to_date = DataProvider.tomorrow().strftime('%d/%m/%Y')
        df = self._download_data(asset["type"], asset["symbol"], asset["name"], asset["country"], from_date, to_date)

        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    @cachetools.func.ttl_cache(maxsize=1, ttl=60 * 60 * 24)
    def update_symbols(self, **kwargs):
        self._build_index()

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

    def update_quotes(self, **kwargs):
        max_date = DataProvider.tomorrow().strftime('%d/%m/%Y')
        max_errors_in_a_row = kwargs['max_errors'] if 'max_errors' in kwargs else 25
        exception_count = 0

        with self.engine.connect() as con:
            if "resume" in kwargs and kwargs["resume"]:
                sql = f"""
                    SELECT symbol, type, country, name, date('1970-01-02') as min_date 
                      FROM {DataProvider.symbols_table_name} s
                     WHERE NOT EXISTS (SELECT * FROM {DataProvider.time_series_table_name} t 
                                        WHERE s.symbol = t.symbol
                                          AND s.type = t.type
                                          AND s.country = t.country LIMIT 1)
                """
                if kwargs["resume"] == "all":
                    sql += f"""
                         UNION ALL
                        SELECT s.symbol, s.type, s.country, s.name, date(max(t.date), '-5 day') as min_date
                          FROM {DataProvider.symbols_table_name} s
                          JOIN {DataProvider.time_series_table_name} t ON s.country = t.country 
                                                                      AND s.symbol = t.symbol 
                                                                      AND s.type = t.type
                         GROUP BY s.symbol, s.type, s.country, s.name
                    """

                assets = list(con.execute(sql))
                _log.info(f"\n======\nresume download of {len(assets)} assets\n======\n")
            else:
                assets = list(con.execute(f'SELECT * FROM {DataProvider.symbols_table_name}'))

        for asset in assets:
            asset_type = asset["type"]
            from_date = (datetime.strptime(asset["min_date"], '%Y-%m-%d') if "min_date" in asset else Investing.min_date).strftime('%d/%m/%Y')

            try:
                df = self._download_data(asset_type, asset["symbol"], asset["name"], asset["country"], from_date, max_date)

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
                _log.error(f"unlisted symbol: {asset} {ie}")
            except Exception as e:
                exception_count += 1
                _log.error(f"error for symbol: {asset} {e}")

                if exception_count >= max_errors_in_a_row:
                    raise ValueError("Something is really wrong!")

    @cachetools.func.ttl_cache(maxsize=100, ttl=10 * 60)
    def _download_data(self, asset_type, symbol, name, country, from_date, max_date):
        if asset_type == "BOND":
            # name == symbol
            df = ip.get_bond_historical_data(symbol, from_date=from_date, to_date=max_date)
        elif asset_type == "CERT":
            df = ip.get_certificate_historical_data(name, country=country, from_date=from_date, to_date=max_date)
        elif asset_type == "CRYPTO":
            df = ip.get_crypto_historical_data(name, from_date=from_date, to_date=max_date)
        elif asset_type == "COMM":
            df = ip.get_commodity_historical_data(symbol, from_date=from_date, to_date=max_date)
        elif asset_type == "ETF":
            df = ip.get_etf_historical_data(name, country=country, from_date=from_date, to_date=max_date)
        elif asset_type == "FUND":
            df = ip.get_fund_historical_data(name, country=country, from_date=from_date, to_date=max_date)
        elif asset_type == "FX":
            df = ip.get_currency_cross_historical_data(symbol, from_date=from_date, to_date=max_date)
        elif asset_type == "INDEX":
            df = ip.get_index_historical_data(name, country=country, from_date=from_date, to_date=max_date)
        elif asset_type == "STOCK":
            df = ip.get_stock_historical_data(symbol, country=country, from_date=from_date, to_date=max_date)

        return df


# for an initial data load we call it directly from here...
if __name__ == '__main__':
    # Investing().update_symbols()
    # check error: 2020-10-08 11:12:09 | ERROR     | pangres     | investing:update_quotes:167 - error for symbol: ('SR', 'STOCK', 'canada', 'Strategic Resources', '1970-01-02') 'float' object has no attribute 'lower'
    Investing().update_quotes(resume='all', max_errors=1500)
