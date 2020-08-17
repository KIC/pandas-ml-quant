from unittest import TestCase

import torch as T
import torch.nn as nn

from pandas_ml_quant import np, PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.model.agent import ReinforceAgent, Network
from pandas_ml_quant_rl.model.environments.multi_symbol_environment import RandomAssetEnv
from pandas_ml_quant_rl.model.strategies.discrete import BuyOpenSellCloseSellOpenBuyClose
from pandas_ml_quant_rl_test.config import load_symbol
from pandas_ml_utils.pytorch import Reshape


class TestAgents(TestCase):

    def test_reinforce_agent(self):
        env = RandomAssetEnv(
            PostProcessedFeaturesAndLabels(
                features=[
                    lambda df: df.ta.candle_category().ta.one_hot_encode_discrete(offset=-15, nr_of_classes=30)
                ],
                labels=[],
                feature_post_processor=[
                    lambda df: df.ta.rnn(10)
                ],
            ),
            ["SPY", "GLD"],
            strategy=BuyOpenSellCloseSellOpenBuyClose(),
            pct_train_data=0.8,
            max_steps=50,
            min_training_samples=50,
            use_cache=FileCache('/tmp/agent.test.dada.hd5', load_symbol),
        )

        class Net(Network):

            def __init__(self):
                super().__init__()
                flattened_obs_size = np.array(30 * 10).prod()

                self.net = nn.Sequential(
                    Reshape(flattened_obs_size),
                    nn.Linear(flattened_obs_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3)
                )

            def estimate(self, feature_state: T.Tensor, portfolio_state: T.Tensor):
                return self.net(feature_state)


        agent = ReinforceAgent(Net())
        agent.fit(env)


