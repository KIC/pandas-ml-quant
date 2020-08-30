from unittest import TestCase

from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.environments import RandomAssetEnv
from pandas_ml_quant_rl.model.strategies.discrete import BuyOpenSellCloseSellOpenBuyClose
from pandas_ml_quant_rl_test.config import load_symbol
from pandas_ml_utils import FeaturesAndLabels

env = RandomAssetEnv(
            PostProcessedFeaturesAndLabels(
                features = [
                    lambda df: df["Close"].ta.log_returns()
                ],
                labels=[],
                feature_post_processor=[
                    lambda df: df.ta.rnn(60)
                ],
            ),
            ["SPY", "GLD"],
            BuyOpenSellCloseSellOpenBuyClose(),
            pct_train_data=0.8,
            max_steps=10,
            use_cache=FileCache('/tmp/lalala.hd5', load_symbol)
        )


class TestEnvironments(TestCase):

    def test_multi_symbol_env_reset(self):

        _, fresh_train = env.as_train()
        print(fresh_train)
        print(len(env._features), env._state_idx)

        _, fresh_test = env.as_test()
        print(fresh_test)
        print(len(env._features), env._state_idx)

        _, fresh_predict = env.as_predict()
        print(fresh_predict)
        print(len(env._features), env._state_idx)

        print(env.observation_space)

    def test_multi_symbol_env_action(self):

        for _ in range(2):
            print(env.step(env.strategy.sample_action()))

    def test_sample_observation_space(self):
        print(env.observation_space)
        print(env.observation_space.sample())

    def test_multi_feature_env(self):
        env = RandomAssetEnv(
            FeaturesAndLabels(
                features=(
                    [
                        lambda df: df.ta.candle_category().ta.one_hot_encode_discrete(offset=-15, nr_of_classes=30).ta.rnn(10)
                    ],
                    [
                        lambda df: df["Close"].ta.sma().ta.rnn(5)
                    ]
                ),
                labels=[],
            ),
            ["SPY", "GLD"],
            strategy=BuyOpenSellCloseSellOpenBuyClose()
        )

        features, portfolio = env.observation_space.sample()
        self.assertIsInstance(features, tuple)
        self.assertEqual((10, 1, 30), features[0].shape)
        self.assertEqual((5, 1), features[1].shape)
