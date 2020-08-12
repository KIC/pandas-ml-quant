import os
from pandas_ml_common import pd


def load_symbol(symbol):
    return pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", f"{symbol}.csv"),
        index_col='Date',
        parse_dates=True
    )

