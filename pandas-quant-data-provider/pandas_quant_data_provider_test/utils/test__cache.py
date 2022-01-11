from unittest import TestCase
import numpy as np
from pandas_quant_data_provider.utils.cache import requests_cache, remove_cached_response


class TestCacheUtil(TestCase):

    def test__cache_keys(self):
        @requests_cache(return_was_cached_flag=True)
        def testing(url):
            return url + "22"

        url = f"http://localhost/{np.random.randint(0, 9999999999)}"

        try:
            res, was_cached = testing(url)
            self.assertEqual(url + "22", res)
            self.assertFalse(was_cached)

            res, was_cached = testing(url)
            self.assertEqual(url + "22", res)
            self.assertTrue(was_cached)
        finally:
            remove_cached_response(url)