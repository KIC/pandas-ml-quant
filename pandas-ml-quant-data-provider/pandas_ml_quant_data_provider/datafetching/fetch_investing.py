import logging

import cachetools.func
import cachetools.func
import investpy

from pandas_ml_common.utils import inner_join, add_multi_index

_log = logging.getLogger(__name__)

START_DATE = '01/01/1960'
STOP_DATE = '31/12/2029'

ASSET_CATEGORY = {
    "bond": lambda s, c: investpy.get_bond_historical_data(bond=s, from_date=START_DATE, to_date=STOP_DATE),
    "certificate": lambda s, c: investpy.get_certificate_historical_data(certificate=s, country=c, from_date=START_DATE, to_date=STOP_DATE),
    "commodity": lambda s, c: investpy.get_commodity_historical_data(commodity=s, from_date=START_DATE, to_date=STOP_DATE),
    "crypto": lambda s, c: investpy.get_crypto_historical_data(crypto=s, from_date=START_DATE, to_date=STOP_DATE),
    "currency": lambda s, c: investpy.get_currency_cross_historical_data(currency_cross=s, from_date=START_DATE, to_date=STOP_DATE),
    "etf": lambda s, c: investpy.get_etf_historical_data(etf=s, country=c, from_date=START_DATE, to_date=STOP_DATE),
    "fund": lambda s, c: investpy.get_fund_historical_data(fund=s, country=c, from_date=START_DATE, to_date=STOP_DATE),
    "index": lambda s, c: investpy.get_index_historical_data(index=s, country=c, from_date=START_DATE, to_date=STOP_DATE),
    "stock": lambda s, c: investpy.get_stock_historical_data(stock=s, country=c, from_date=START_DATE, to_date=STOP_DATE),
}


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_investing(*args: str, multi_index: bool = False, **kwargs):
    df = None

    for compound_symbol in args:
        details = compound_symbol.split("::")
        asset_class, symbol, country = details if len(details) >= 3 else (*details, None)
        _df = _fetch_investing(asset_class, symbol, country)

        if multi_index:
            _df = add_multi_index(_df, symbol, True)

        if df is None:
            df = _df.add_prefix(symbol) if len(args) > 1 and not multi_index else _df
        else:
            df = inner_join(df, _df, prefix='' if multi_index else symbol, force_multi_index=multi_index)

    return df


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def _fetch_investing(asset_category, symbol, country):
    try:
        return ASSET_CATEGORY[asset_category](symbol, country)
    except IndexError as ie:
        _log.warning(f"{ie}, {asset_category}, {symbol}, {country}")

