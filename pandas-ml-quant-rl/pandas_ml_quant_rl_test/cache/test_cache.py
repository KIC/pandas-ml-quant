from unittest import TestCase

from pandas_ml_quant import pd, np
from pandas_ml_quant_rl.cache import NoCache, FileCache
from pandas_ml_quant_rl.cache.mem_cache import MemCache
from pandas_ml_quant_rl_test.config import load_symbol
from pandas_ml_utils import FeaturesAndLabels


class TestFileCache(TestCase):

    def test_file_cache(self):
        fl = FeaturesAndLabels(
            features=[lambda df: df["Close"].ta.sma()]
        )

        cache1 = NoCache(load_symbol)
        cache2 = FileCache('/tmp/cache.test', load_symbol)

        for i in range(2):
            pd.testing.assert_frame_equal(
                cache1.get_data_or_fetch("SPY"),
                cache2.get_data_or_fetch("SPY")
            )

            np.testing.assert_almost_equal(
                cache1.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0],
                cache2.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0]
            )

    def test_file_cache_with_multi_frame(self):
        fl = FeaturesAndLabels(
            features=(
                [lambda df: df["Close"].ta.sma()],
                [lambda df: df["Close"].ta.macd()]
            )
        )

        cache1 = NoCache(load_symbol)
        cache2 = FileCache('/tmp/cache.test', load_symbol)

        for i in range(2):
            pd.testing.assert_frame_equal(
                cache1.get_data_or_fetch("SPY"),
                cache2.get_data_or_fetch("SPY")
            )

            np.testing.assert_almost_equal(
                cache1.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0][0],
                cache2.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0][0]
            )

    def test_mem_cache(self):
        fl = FeaturesAndLabels(
            features=[lambda df: df["Close"].ta.sma()]
        )

        cache1 = NoCache(load_symbol)
        cache2 = MemCache(load_symbol)

        for i in range(2):
            pd.testing.assert_frame_equal(
                cache1.get_data_or_fetch("SPY"),
                cache2.get_data_or_fetch("SPY")
            )

            np.testing.assert_almost_equal(
                cache1.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0],
                cache2.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0]
            )

    def test_mem_cache_with_multi_frame(self):
        fl = FeaturesAndLabels(
            features=(
                [lambda df: df["Close"].ta.sma()],
                [lambda df: df["Close"].ta.macd()]
            )
        )

        cache1 = NoCache(load_symbol)
        cache2 = MemCache(load_symbol)

        for i in range(2):
            pd.testing.assert_frame_equal(
                cache1.get_data_or_fetch("SPY"),
                cache2.get_data_or_fetch("SPY")
            )

            np.testing.assert_almost_equal(
                cache1.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0][0],
                cache2.get_feature_frames_or_fetch(cache1.get_data_or_fetch("SPY"), "SPY", fl)[0][0]
            )
