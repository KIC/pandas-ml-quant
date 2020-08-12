from unittest import TestCase

import gym.spaces as spaces

from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.model.environments.multi_symbol_environment import RandomAssetEnv
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
            spaces.Discrete(2),
            load_symbol,
            0.8
        )


class TestEnvironments(TestCase):

    def test_multi_symbol_env_reset(self):
        fresh_train = env.as_train().reset()
        print(fresh_train)
        print(len(env._features), len(env._labels), env._state_idx)

        fresh_test = env.as_test().reset()
        print(fresh_test)
        print(len(env._features), len(env._labels), env._state_idx)

        fresh_predict = env.as_predict().reset()
        print(fresh_predict)
        print(len(env._features), len(env._labels), env._state_idx)

        print(env.observation_space)

    def test_multi_symbol_env_action(self):
        def step(s, a, l):
            print(s.shape, a, l[0].shape)
            return 0, False

        env.action_executor = step
        env.as_train().reset()

        for act_space in [spaces.Discrete(3), spaces.Box(low=-1, high=1, shape=(2,))]:
            done = False
            while not done:
                env.action_space = act_space
                s, r, done, _ = env.step(env.action_space.sample())

            env.reset()

