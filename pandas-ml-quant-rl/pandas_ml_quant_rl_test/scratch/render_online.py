from time import sleep
from unittest import TestCase

import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np

from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.model.environments.multi_symbol_environment import RandomAssetEnv
from pandas_ml_quant_rl.renderer import CandleStickRenderer, OnlineRenderer
from pandas_ml_quant_rl_test.config import load_symbol


def render_from_environment():

    env = RandomAssetEnv(
        PostProcessedFeaturesAndLabels(
            features=[
                lambda df: df["Close"].ta.log_returns()
            ],
            labels=[
                lambda df: df["Close"].ta.log_returns().shift(-1)
            ],
            feature_post_processor=[
                lambda df: df.ta.rnn(60)
            ],
        ),
        ["SPY", "GLD"],
        spaces.Discrete(3),
        pct_train_data=0.8,
        max_steps=10,
        use_cache=FileCache('/tmp/lalala.hd5', load_symbol),
        reward_provider=lambda *x: (np.random.random(1) - 0.5, False),
        renderer=OnlineRenderer(
            lambda: CandleStickRenderer
                (action_mapping=[(1, 'buy', 'Open'), (1, 'sell', 'Close'), (2, 'sell', 'Open'), (2, 'buy', 'Close')])
        )
    )

    env.render()
    for _ in range(2):
        print("play episode")
        env.reset()
        done = False
        while not done:
            s, r, done, _ = env.step(env.action_space.sample())
            print(s.index, done)

    sleep(2)
    env.renderer.stop()


if __name__ == '__main__':
    render_from_environment()
