import piphyperd

try:
    print(piphyperd.PipHyperd("-U").install("yfinance"))
finally:
    from .yfinance_provider import YahooSymbol


def clear_cache():
    from .yfinance_provider import _fetch_hist
    _fetch_hist.clear_cache()
