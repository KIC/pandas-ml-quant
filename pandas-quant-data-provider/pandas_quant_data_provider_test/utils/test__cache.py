from datetime import datetime
from time import sleep
from unittest import TestCase

import numpy as np

from pandas_quant_data_provider.utils.cache import requests_cache, remove_cached_response


class TestCacheUtil(TestCase):

    url = f"http://localhost/{np.random.randint(0, 9999999999)}"

    @staticmethod
    @requests_cache(return_was_cached_flag=True)
    def request_testing(url, **kwargs):
        return url + "22"

    def test__cache_keys(self):
        try:
            res, was_cached = TestCacheUtil.request_testing(TestCacheUtil.url)
            self.assertEqual(TestCacheUtil.url + "22", res)
            self.assertFalse(was_cached)

            res, was_cached = TestCacheUtil.request_testing(TestCacheUtil.url)
            self.assertEqual(TestCacheUtil.url + "22", res)
            self.assertTrue(was_cached)
        finally:
            remove_cached_response(TestCacheUtil.url)

    def test__cache_ttl(self):
        try:
            if datetime.utcnow().second == 0:
                sleep(0.01)

            res, was_cached = TestCacheUtil.request_testing(TestCacheUtil.url, caching='__testing__')
            self.assertEqual(TestCacheUtil.url + "22", res)
            self.assertFalse(was_cached)

            res, was_cached = TestCacheUtil.request_testing(TestCacheUtil.url, caching='__testing__')
            self.assertEqual(TestCacheUtil.url + "22", res)
            self.assertTrue(was_cached)

            sleep(3)
            res, was_cached = TestCacheUtil.request_testing(TestCacheUtil.url, caching='__testing__')
            self.assertEqual(TestCacheUtil.url + "22", res)
            self.assertFalse(was_cached)
        finally:
            remove_cached_response(TestCacheUtil.url)
