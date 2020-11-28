from pandas_ml_quant_data_provider.datafetching import fetch_yahoo, fetch_investing, fetch_cryptocompare_daily, \
    fetch_cryptocompare_hourly, fetch_fred

YAHOO = fetch_yahoo
INVESTING = fetch_investing
CRYPTO_COMPARE = fetch_cryptocompare_daily
CRYPTO_COMPARE_HOURLY = fetch_cryptocompare_hourly
FRED = fetch_fred

PROVIDER_MAP = {
    "yahoo": YAHOO,
    "investing": INVESTING,
    "cryptocompare": CRYPTO_COMPARE,
    "cryptocompare_daily": CRYPTO_COMPARE,
    "cryptocompare_hourly": CRYPTO_COMPARE_HOURLY,
    "fred": FRED
}