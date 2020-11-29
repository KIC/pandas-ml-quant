from .yfinance_provider import fetch_yahoo, _fetch_hist


def clear_cache():
    _fetch_hist.clear_cache()