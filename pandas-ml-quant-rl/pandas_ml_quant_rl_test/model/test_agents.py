from unittest import TestCase

import torch as T
import torch.nn as nn
from torch.functional import F

from pandas_ml_quant_rl.model.strategies.discrete import LongOnly
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_quant import np, PostProcessedFeaturesAndLabels
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.model.agent import ReinforceAgent, PolicyNetwork, DQNAgent
from pandas_ml_quant_rl.model.environments.multi_symbol_environment import RandomAssetEnv
from pandas_ml_quant_rl.model.strategies import BuyOpenSellCloseSellOpenBuyClose, LongShortSwing
from pandas_ml_quant_rl_test.config import load_symbol
from pandas_ml_utils.pytorch import Reshape


env = RandomAssetEnv(
    FeaturesAndLabels(
        features=
            [
                lambda df: df.ta.candle_category().ta.one_hot_encode_discrete(offset=-15, nr_of_classes=30).ta.rnn(10)
            ],

        labels=[],
    ),
    ["SPY", "GLD"],
    strategy=LongOnly(),
    pct_train_data=0.8,
    max_steps=50,
    min_training_samples=50,
    use_cache=FileCache('/tmp/agent.test.dada.hd5', load_symbol),
)


class TestAgents(TestCase):

    def test_reinforce_agent(self):

        class Net(PolicyNetwork):

            def __init__(self, nr_features, nr_portfolio_states, nr_actions):
                super().__init__()
                self.input1 = nn.Sequential(Reshape(nr_features), nn.Linear(nr_features, 32))
                self.input2 = nn.Sequential(Reshape(nr_portfolio_states), nn.Linear(nr_portfolio_states, 2))
                self.output = nn.Linear(34, nr_actions)
                self.sm = nn.Softmax()

            def estimate_action(self, feature_state: T.Tensor, portfolio_state: T.Tensor):
                x1 = F.relu(self.input1(feature_state))
                x2 = F.tanh(self.input2(portfolio_state))
                return self.sm(self.output(T.cat([x1, x2], dim=1)))

        agent = ReinforceAgent(
            Net(nr_features=30*10, nr_portfolio_states=1, nr_actions=env.strategy.action_space.n),
            exit_criteria=lambda reward, cnt: reward > 1 or cnt > 10
        )

        agent.fit(env)

    def test_dqn_agent(self):
        class Net(PolicyNetwork):

            def __init__(self, nr_features, nr_portfolio_states, nr_actions):
                super().__init__()
                self.input1 = nn.Sequential(Reshape(nr_features), nn.Linear(nr_features, 32))
                self.input2 = nn.Sequential(Reshape(nr_portfolio_states), nn.Linear(nr_portfolio_states, 2))
                self.output = nn.Linear(34, nr_actions)

            def estimate_action(self, feature_state: T.Tensor, portfolio_state: T.Tensor):
                x1 = F.relu(self.input1(feature_state))
                x2 = F.tanh(self.input2(portfolio_state))
                return self.output(T.cat([x1, x2], dim=1))

        agent = DQNAgent(
            lambda: Net(nr_features=30*10, nr_portfolio_states=1, nr_actions=env.strategy.action_space.n),
            exit_criteria=lambda reward, cnt: reward > 1 or cnt > 10
        )

        agent.fit(env)

