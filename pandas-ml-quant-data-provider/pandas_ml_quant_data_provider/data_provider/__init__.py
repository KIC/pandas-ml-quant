from .yfinance_provider import fetch_yahoo


def clear_cache():
    fetch_yahoo.clear_cache()