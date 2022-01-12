import logging

from diskcache import Index, Cache
from typing import Callable
from os import path
from datetime import datetime, timedelta

CACHE_DIR = path.join(path.expanduser("~"), ".cache", "pandas_quant_data_provider")
PERSISTENT_CACHE = Index(CACHE_DIR)
CACHE = Cache(CACHE_DIR)

logger = logging.getLogger(__name__)
logger.info(f"using persistent requests cache stored at {PERSISTENT_CACHE.directory}")


class Duration(object):

    forever = 'forever'
    day = 'day'
    hour = 'hour'
    minute = 'minute'


def requests_cache(return_was_cached_flag=False) -> Callable:
    def decorated(func: Callable):
        def wrapped(url, *args, **kwargs):
            caching_duration = kwargs.get("caching", Duration.forever)
            cache = PERSISTENT_CACHE
            caching_kwargs = dict()

            if caching_duration is None or kwargs.get("no_cache", False):
                res = func(url, *args, **kwargs)
                return (res, False) if return_was_cached_flag else res
            elif caching_duration == Duration.day:
                evict_at = (datetime.utcnow() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                caching_kwargs["expire"] = (evict_at - datetime.utcnow()).seconds
                cache = CACHE
            elif caching_duration == Duration.hour:
                evict_at = (datetime.utcnow() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                caching_kwargs["expire"] = (evict_at - datetime.utcnow()).seconds
                cache = CACHE
            elif caching_duration == Duration.minute:
                evict_at = (datetime.utcnow() + timedelta(minutes=1)).replace(second=0, microsecond=0)
                caching_kwargs["expire"] = (evict_at - datetime.utcnow()).seconds
                cache = CACHE
            elif caching_duration == '__testing__':
                caching_kwargs["expire"] = 2
                cache = CACHE

            cached = cache.get(url, default=None)

            if cached is None:
                value = func(url, *args, **kwargs)
                if caching_kwargs:
                    cache.set(url, value, **caching_kwargs)
                else:
                    cache[url] = value

                return (value, False) if return_was_cached_flag else value
            else:
                return (cached, True) if return_was_cached_flag else cached

        return wrapped

    return decorated


def remove_cached_response(url):
    try:
        del PERSISTENT_CACHE[url]
    except Exception as e:
        pass
