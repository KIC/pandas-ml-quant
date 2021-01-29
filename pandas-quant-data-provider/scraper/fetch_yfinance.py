import logging
import os

import pandas as pd

import pandas_quant_data_provider as qd
from pandas_ml_common import serialize
from scraper.dolting import start_server_and_get_engine, dolt_commit
from concurrent.futures import ThreadPoolExecutor

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

            return symbol, df
        except Exception as e:
            _log.warning(f'failed quote download for {symbol}: {e}')
            return symbol, None

    engine = start_server_and_get_engine()
    #symbols = {row[0]: row[1] for row in engine.execute("SELECT o.symbol, coalesce(max(y.eod), '1900-01-06') - INTERVAL 5 DAY FROM optionable o LEFT OUTER JOIN yahoo y on y.symbol = o.symbol GROUP BY o.symbol")}
    symbols = {row[0]: row[1] for row in engine.execute("SELECT symbol, cast('1900-01-01' AS DATE) FROM optionable ORDER BY symbol")}

    # clean up file
    if os.path.exists('/tmp/quotes.csv'):
        os.remove('/tmp/quotes.csv')

    with ThreadPoolExecutor(3) as executor:
        for running_task in executor.map(fetch, symbols.keys(), symbols.values()):
            try:
                symbol, df = running_task
                if df is not None:
                    df.to_hdf('/tmp/quotes.hdf', symbol)
            except Exception as e:
                _log.error(f"something failed", e)


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