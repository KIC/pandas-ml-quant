import piphyperd

try:
    print(piphyperd.PipHyperd("-U").install("yfinance"))
    from .yfinance_provider import fetch_yahoo
finally:
    pass


def clear_cache():
    from .yfinance_provider import _fetch_hist
    _fetch_hist.clear_cache()