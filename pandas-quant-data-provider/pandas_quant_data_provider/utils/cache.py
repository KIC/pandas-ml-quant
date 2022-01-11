import logging

from diskcache import Index, Cache
from typing import Callable
from os import path

CACHE_DIR = path.join(path.expanduser("~"), ".cache", "pandas_quant_data_provider")
CACHE = Index(CACHE_DIR)

logger = logging.getLogger(__name__)
logger.info(f"using persistent requests cache stored at {CACHE.directory}")


def requests_cache(return_was_cached_flag=False) -> Callable:
    def decorated(func: Callable):
        def wrapped(url, *args, **kwargs):
            if kwargs.get("no_cache", False):
                res = func(url, *args, **kwargs)
                return (res, False) if return_was_cached_flag else res

            cached = CACHE.get(url, default=None)

            if cached is None:
                value = func(url, *args, **kwargs)
                CACHE[url] = value
                return (value, False) if return_was_cached_flag else value
            else:
                return (cached, True) if return_was_cached_flag else cached

        return wrapped

    return decorated


def remove_cached_response(url):
    del CACHE[url]
