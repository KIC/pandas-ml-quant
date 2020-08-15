from unittest import TestCase

import gym.spaces as spaces

from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.model.environments.multi_symbol_environment import RandomAssetEnv
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.model.strategies.discrete import BuyOpenSellCloseSellOpenBuyClose

from pandas_ml_quant_rl_test.config import load_symbol

env = RandomAssetEnv(
            PostProcessedFeaturesAndLabels(
                features = [
                    lambda df: df["Close"].ta.log_returns()
                ],
                labels = [
                    lambda df: df["Close"].ta.log_returns().shift(-1)
                ],
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
        print(len(env._features), len(env._labels), env._state_idx)

        _, fresh_test = env.as_test()
        print(fresh_test)
        print(len(env._features), len(env._labels), env._state_idx)

        _, fresh_predict = env.as_predict()
        print(fresh_predict)
        print(len(env._features), len(env._labels), env._state_idx)

        print(env.observation_space)

    def test_multi_symbol_env_action(self):

        for _ in range(2):
            print(env.step(env.sample_action()))

