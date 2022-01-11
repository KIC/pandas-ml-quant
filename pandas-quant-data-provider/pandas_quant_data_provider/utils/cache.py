import logging

from diskcache import Index, Cache
from typing import Callable
from os import path

CACHE_DIR = path.join(path.expanduser("~"), ".cache", "pandas_quant_data_provider")
CACHE = Index(CACHE_DIR)

logger = logging.getLogger(__name__)
logger.info(f"using persistent requests cache stored at {CACHE.directory}")


def requests_cache(func: Callable) -> Callable:
    def wrapped(url, *args, **kwargs):
        if kwargs.get("no_cache", False):
            return func(url, *args, **kwargs)

        cached = CACHE.get(url, default=None)

        if cached is None:
            value = func(url, *args, **kwargs)
            CACHE[url] = value
            return value
        else:
            return cached

    return wrapped
