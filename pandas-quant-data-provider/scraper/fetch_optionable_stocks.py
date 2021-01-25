import logging
import os
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup
from cachier import cachier

from pandas_ml_common.utils import call_silent
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_quant_data_provider.data_provider.time_utils import time_until_end_of_day

_log = logging.getLogger(__name__)
log_once = LogOnce()


def download_optionable_stocks_and_sectors(sanitize_columns=True):
    # fetch optionable stocks

    df = _fetch_optionable_symbols()
    _log.info(f'fetched {len(df)} optionable symbols: {df.index.to_list()}')

    # add sector and industry to the data frame
    df_si = df.apply(
        lambda row: call_silent(lambda: _get_sector_industry(str(row.name)), log=True, log_ctx=f"get industry {row.name}"),
        result_type='expand',
        axis=1
    )
    df_si.columns = ["Sector", "Industry"]
    df = df.join(df_si)

    # make columns db compatible
    if sanitize_columns:
        df.columns = [col.lower().replace(" ", "_").replace("__", "_") for col in df.columns]

    return df


@cachier(stale_after=time_until_end_of_day())
def _fetch_optionable_symbols():
    # download optionable symbols as frames
    pages = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z IND ETF".split(" ")
    def download_df(page):
        df = pd.read_html(f'https://www.poweropt.com/optionable.asp?fl={page}', attrs={'id': 'example'})[0]
        df["Type"] = "STOCK" if len(page) <= 1 else page
        return df

    df = pd.concat([download_df(page) for page in pages], axis=0)

    # extract symbol
    symbol = df['Company  Name'].str.extract(r'^\(([^\(\)]+)\)').iloc[:, 0].rename("Symbol")
    symbol = symbol.str.replace("$", "^")

    if len(symbol[symbol.isnull().values]) > 0:
        raise ValueError(f"Could not extract Symbol:\n{df[symbol.isnull().values]}")

    # make symbol the index and add drop unnecessary data
    df.index = symbol
    df = df.drop_duplicates(keep='last').sort_index()
    df['Last Update'] = pd.Timestamp.now()
    df = df.drop(["Quick  Find", "Option  Chain", "Details"], axis=1)
    return df


@cachier()
def _get_sector_industry(symbol):
    log_once.log(symbol[0], _log.info, f"fetching industry for {symbol[0]}")
    _log.debug(f"fetching industry for {symbol}")

    resp = requests.get(f"https://finance.yahoo.com/quote/{symbol}/profile")
    html = BeautifulSoup(resp.text, features="lxml")

    try:
        sector = html.select('span[class="Fw(600)"]')[0].text
    except:
        sector = html.select('span[class="Fl(end)"]')[0].text

    try:
        industry = html.select('span[class="Fw(600)"]')[1].text
    except:
        industry = None

    return sector, industry


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    pd.set_option('display.max_columns', None)

    filename = sys.argv[1] if len(sys.argv) > 1 else '/tmp/optionable.csv'
    df = download_optionable_stocks_and_sectors(True)
    df.to_csv('/tmp/optionable.csv')

    if os.path.isdir(filename) and os.path.exists(os.path.join(filename, ".dolt")):
        from scraper.dolting import start_server_and_get_engine, mysql_replace_into, dolt_commit
        try:
            _log.info(f"insert {len(df)} rows into dolt versioned database")
            df.to_sql("optionable", start_server_and_get_engine(os.path.abspath(filename)), if_exists='append', method=mysql_replace_into)
            dolt_commit("optionable", "update optionable")
        except Exception as e:
            _log.error(f'failed sql insert! A recovery file can be found at: /tmp/optionable.csv', e)
            df.to_csv('/tmp/optionable.csv')
    else:
        df.to_csv(filename)

    _log.info(f"updating {filename} is done!")


# in airflow we could make this a one liner:
# download_optionable_stocks_and_sectors(True).to_sql("optionable", start_server_and_get_engine(os.path.abspath(filename)), if_exists='append', method=mysql_replace_into)
# > dolt commit > dolt push
