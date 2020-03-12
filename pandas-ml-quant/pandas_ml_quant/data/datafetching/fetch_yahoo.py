import logging
import traceback

import cachetools.func

from pandas_ml_common import pd
from pandas_ml_common.utils import merge_kwargs, inner_join


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_yahoo(*args: str, period: str = 'max', multi_index: bool = False, **kwargs: str):
    df = None

    if len(args) == 1:
        df = __download_yahoo_data(args[0], period)
    else:
        # convert args to kwargs
        if len(args) > 0:
            kwargs = merge_kwargs({arg: arg for arg in args}, kwargs)

        for k, v in kwargs.items():
            px = f'{k}_'
            df_ = __download_yahoo_data(v, period)

            if multi_index:
                df_.columns = pd.MultiIndex.from_product([[k], df_.columns])

                if df is None:
                    df = df_
                else:
                    df = inner_join(df, df_)
            else:
                if df is None:
                    df = df_.add_prefix(px)
                else:
                    df = inner_join(df, df_, prefix=px)

    # print some statistics
    if df is None:
        logging.warning("nothing downloaded")
    else:
        logging.info(f'number of rows for joined dataframe = {len(df)}, from {df.index[0]} to {df.index[-1]}')

    return df


def __download_yahoo_data(symbol, period):
    import yfinance as yf
    df = None

    # bloody skew index does not have any data on yahoo
    if symbol == '^SKEW':
        df = pd.read_csv('http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/skewdailyprices.csv',
                          skiprows=1,
                          parse_dates=True,
                          index_col='Date') \
            .drop(['Unnamed: 2', 'Unnamed: 3'], axis=1)
    else:
        ticker = yf.Ticker(symbol)
        try:
            # first try to append the most recent data
            df = ticker.history(period="1d", interval='1d')[-1:].combine_first(ticker.history(period=period))
        except IOError:
            traceback.print_exc()
            logging.warning(
                'failed to add yf.Ticker({v}).history(period="1d", interval="1d")[-1:] fallback to hist only!')
            df = ticker.history(period=period)

    # print some statistics
    logging.info(f'number of rows for {symbol} = {len(df)}, from {df.index[0]} to {df.index[-1]} period={period}')

    return df
