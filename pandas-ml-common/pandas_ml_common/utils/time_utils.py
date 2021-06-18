from datetime import datetime, timedelta
from typing import Union, Tuple

import pandas as pd


def seconds_since_midnight():
    return int((datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())


def parse_timestamp(timestamp: Union[pd.Timestamp, str, Tuple[str, str]],):
    return (timestamp.tz_convert('UTC') if is_timezone_aware(timestamp) else timestamp.tz_localize('UTC')) if isinstance(timestamp, pd.Timestamp) else \
        pd.to_datetime(timestamp[0], format=timestamp[1], utc=True) if isinstance(timestamp, tuple) else pd.to_datetime(timestamp, utc=True)


def is_timezone_aware(tst: pd.Timestamp):
    return tst.tzinfo is not None and tst.tzinfo.utcoffset(tst) is not None


def min_timestamp():
    return pd.Timestamp.min.tz_localize('UTC')


def make_timeindex(from_tst: pd.Timestamp, nr_of_timesteps: int, only_weekdays=False, include_start_date=False):
    tst = from_tst
    forecast_tst = []

    while len(forecast_tst) < nr_of_timesteps:
        tst += timedelta(days=1)

        if only_weekdays and tst.isoweekday() > 5:
            continue

        forecast_tst.append(tst)

    return ([from_tst] if include_start_date else []) + forecast_tst
