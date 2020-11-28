import pandas as pd


def read_ts_csv(filename, **kwargs):
    return pd.read_csv(filename, parse_dates=True, index_col='Date', **kwargs)