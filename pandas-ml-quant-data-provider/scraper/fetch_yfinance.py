import logging
import os
import pandas as pd
import pandas_ml_quant_data_provider as qd
from scraper.dolting import start_server_and_get_engine, mysql_replace_into, dolt_commit, import_df

_log = logging.getLogger(__name__)


def download_quotes():
    def fetch(symbol, date='1990-01-01'):
        assert isinstance(symbol, str), f"Expected symbol as string, got {type(symbol)}"
        symbol = symbol.replace('$', '^')
        _log.info(f"fetch quote for >{symbol}<")
        try:
            df = qd.fetch(symbol, start=date)
            df.index = pd.MultiIndex.from_product([[symbol], df.index], names=["symbol", "eod"])
            df.columns = [col.lower().replace(" ", "_").replace("__", "_") for col in df.columns]

            return df
        except Exception as e:
            _log.warning(f'failed quote download for {symbol}: {e}')
            return None

    engine = start_server_and_get_engine()
    #symbols = {row[0]: row[1] for row in engine.execute("SELECT o.symbol, coalesce(max(y.eod), '1900-01-06') - INTERVAL 5 DAY FROM optionable o LEFT OUTER JOIN yahoo y on y.symbol = o.symbol GROUP BY o.symbol")}
    symbols = {row[0]: row[1] for row in engine.execute("SELECT symbol, cast('1900-01-01' AS DATE) FROM optionable ORDER BY symbol")}

    for symbol, date in symbols.items():
        df = fetch(symbol, date)
        if df is not None:
            _log.info(f"update db {len(df)} x {df.columns}")
            try:
                df.to_sql("yahoo", engine, if_exists='append', method=mysql_replace_into)
            except Exception as e:
                _log.error(f"failed to insert data of symbol: {symbol} > {date}: {e}")
                engine = start_server_and_get_engine()


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    download_quotes()
    dolt_commit("yahoo", "update quotes")

# in airflow we could make this a one liner:
# download_quotes() >> dolt commit > dolt push
"""
create table yahoo(
    symbol varchar(100) not null, 
    eod date,
    open double,
    high double,
    low double,
    close double,
    volume double,
    dividends double,
    stock_splits double,
    adj_close double,
    primary key(symbol, eod)
);


"""