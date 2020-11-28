import logging
import os
import pandas as pd
import pandas_ml_quant_data_provider as qd
from scraper.dolting import start_server_and_get_engine, mysql_replace_into, dolt_commit

_log = logging.getLogger(__name__)


def download_quotes():
    def fetch(symbol):
        symbol = symbol.replace('$', '^')
        _log.info(f"fetch quote for {symbol}")
        try:
            df = qd.fetch(symbol)
            df.index = pd.MultiIndex.from_product([[symbol], df.index], names=["symbol", "eod"])
            df.columns = [col.lower().replace(" ", "_").replace("__", "_") for col in df.columns]

            return df
        except Exception as e:
            _log.warning(f'failed quote download for {symbol}: {e}')
            return None

    engine = start_server_and_get_engine()
    symbols = engine.execute("SELECT symbol FROM optionable ORDER BY symbol")

    for symbol in symbols:
        df = fetch(symbol[0])
        if df is not None:
            _log.info(f"update db {len(df)} x {df.columns}")
            df.to_sql("yahoo", engine, if_exists='append', method=mysql_replace_into)


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    download_quotes()
    dolt_commit("yahoo", "update quotes")

# in airflow we could make this a one liner:
# download_quotes() >> dolt commit > dopt push
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
    primary key(symbol, eod)
);
"""