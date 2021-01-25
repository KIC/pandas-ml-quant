# slightly enhanced version of https://github.com/lagerfeuer/cryptocompare/blob/f940fab908a9b79ce9069cc6dea9cc5a3f2e2eee/cryptocompare/cryptocompare.py

import datetime
import logging
import time

import cachetools.func
import requests

_log = logging.getLogger(__name__)


# API
URL_COIN_LIST = 'https://www.cryptocompare.com/api/data/coinlist/'
URL_PRICE = 'https://min-api.cryptocompare.com/data/pricemulti?fsyms={}&tsyms={}'
URL_PRICE_MULTI = 'https://min-api.cryptocompare.com/data/pricemulti?fsyms={}&tsyms={}'
URL_PRICE_MULTI_FULL = 'https://min-api.cryptocompare.com/data/pricemultifull?fsyms={}&tsyms={}'
URL_HIST_PRICE = 'https://min-api.cryptocompare.com/data/pricehistorical?fsym={}&tsyms={}&ts={}&e={}'
URL_HIST_PRICE_DAY = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&allData={}'
URL_HIST_PRICE_HOUR = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&toTs={}'
URL_HIST_PRICE_MINUTE = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}'
URL_AVG = 'https://min-api.cryptocompare.com/data/generateAvg?fsym={}&tsym={}&e={}'
URL_EXCHANGES = 'https://www.cryptocompare.com/api/data/exchanges'

# MAX
MAX_LIMIT = 2000
MAX_INT = 2**31 - 1

# FIELDS
PRICE = 'PRICE'
HIGH = 'HIGH24HOUR'
LOW = 'LOW24HOUR'
VOLUME = 'VOLUME24HOUR'
CHANGE = 'CHANGE24HOUR'
CHANGE_PERCENT = 'CHANGEPCT24HOUR'
MARKETCAP = 'MKTCAP'
DATA = 'Data'
TIME = 'time'

# DEFAULTS
CURR = 'USD'
LIMIT = 1440
###############################################################################


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def query_cryptocompare(url,errorCheck=True):
    try:
        _log.debug(url)
        response = requests.get(url).json()
    except Exception as e:
        _log.error('Error getting coin information. %s' % str(e))
        return None
    if errorCheck and (response.get('Response') == 'Error'):
        _log.error('[ERROR] %s' % response.get('Message'))
        return None
    return response


def format_parameter(parameter):
    if isinstance(parameter, list):
        return ','.join(parameter)
    else:
        return parameter

###############################################################################


def get_coin_list(format=False):
    response = query_cryptocompare(URL_COIN_LIST, False)['Data']
    if format:
        return list(response.keys())
    else:
        return response


# TODO: add option to filter json response according to a list of fields
def get_price(coin, curr=CURR, full=False):
    if full:
        return query_cryptocompare(URL_PRICE_MULTI_FULL.format(format_parameter(coin),
            format_parameter(curr)))
    if isinstance(coin, list):
        return query_cryptocompare(URL_PRICE_MULTI.format(format_parameter(coin),
            format_parameter(curr)))
    else:
        return query_cryptocompare(URL_PRICE.format(coin, format_parameter(curr)))


def get_historical_price(coin, curr=CURR, timestamp=time.time(), exchange='CCCAGG'):
    if isinstance(timestamp, datetime.datetime):
        timestamp = time.mktime(timestamp.timetuple())
    return query_cryptocompare(URL_HIST_PRICE.format(coin, format_parameter(curr),
        int(timestamp), format_parameter(exchange)))


def get_historical_price_day(coin, curr=CURR, limit=LIMIT):
    all_data = "false"

    if limit is None or limit > LIMIT:
        limit = 1
        all_data = "true"

    return query_cryptocompare(URL_HIST_PRICE_DAY.format(coin, format_parameter(curr), limit, all_data))


def get_historical_price_hour(coin, curr=CURR, limit=LIMIT):
    current_ts = int(time.time()) + 60 * 60 * 24 + 1 # to be on the safe side
    if limit is None or limit > LIMIT:
        _log.info("batch download < now")
        youngest_ts = get_historical_price_day(coin, curr, None)["TimeFrom"] / 1000
        data = query_cryptocompare(URL_HIST_PRICE_HOUR.format(coin, format_parameter(curr), MAX_LIMIT, current_ts))
        batch = data

        while True:
            last_ts = min(batch[DATA][0][TIME], batch[DATA][-1][TIME])
            _log.info(f"batch download < {last_ts}")
            batch = query_cryptocompare(URL_HIST_PRICE_HOUR.format(coin, format_parameter(curr), MAX_LIMIT, last_ts - 1))
            if batch is None:
                return data
            else:
                data[DATA] += batch[DATA]
                if len(batch) <= 0 or last_ts < youngest_ts:
                    return data
                elif limit is not None and len(data[DATA]) >= limit:
                    data[DATA] = data[DATA][:limit]
                    return data
    else:
        return query_cryptocompare(URL_HIST_PRICE_HOUR.format(coin, format_parameter(curr), limit, current_ts))


def get_historical_price_minute(coin, curr=CURR, limit=LIMIT):
    return query_cryptocompare(URL_HIST_PRICE_MINUTE.format(coin, format_parameter(curr), limit))


def get_avg(coin, curr=CURR, exchange='CCCAGG'):
    response = query_cryptocompare(URL_AVG.format(coin, curr, format_parameter(exchange)))
    if response:
        return response['RAW']


def get_exchanges():
    response = query_cryptocompare(URL_EXCHANGES)
    if response:
        return response['Data']
