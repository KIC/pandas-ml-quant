from unittest import TestCase

import torch as T
import torch.nn as nn
from torch.functional import F

from pandas_ml_quant_rl.model.strategies.discrete import LongOnly
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_quant_rl.cache import FileCache
from pandas_ml_quant_rl.model.agent import ReinforceAgent, PolicyNetwork, DQNAgent
from pandas_ml_quant_rl.environments import RandomAssetEnv
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

            def get_value_for(self, states, action):
                return self._estimate_action(*states) \
                    .gather(dim=1, index=T.LongTensor(action).to(self.device).unsqueeze(-1))

            def get_state_value(self, states):
                values = self._estimate_action(*states).detach()
                return values.max(dim=1)

            def _estimate_action(self, feature_state: T.Tensor, portfolio_state: T.Tensor):
                x1 = F.relu(self.input1(feature_state))
                x2 = F.tanh(self.input2(portfolio_state))
                return self.output(T.cat([x1, x2], dim=1))

        agent = DQNAgent(
            lambda: Net(nr_features=30*10, nr_portfolio_states=1, nr_actions=env.strategy.action_space.n),
            exit_criteria=lambda reward, cnt: reward > 1 or cnt > 10
        )

        agent.fit(env)

    def test_qn_dqn(self):
        from pandas_ml_quant_rl.model.agent.pytorch.losses import QuantileHuberLoss

        class Net(PolicyNetwork):
            def __init__(self, input_size, n_actions, n_quantiles=32):
                super().__init__()
                self.n_actions = n_actions
                self.n_quantiles = n_quantiles
                self.net = nn.Sequential(
                    Reshape(input_size),
                    nn.Linear(input_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_actions * n_quantiles)
                )

            def get_value_for(self, states, action):
                probs = self._estimate_action(*states)
                index = T.LongTensor(action).unsqueeze(-1).unsqueeze(-1).expand(probs.shape[0], self.n_quantiles, 1)
                return probs.gather(dim=2, index=index)

            def get_state_value(self, states):
                probs = self._estimate_action(*states)

                return (
                    probs \
                        .gather(dim=2, index=T.argmax(probs.mean(dim=1), dim=1, keepdim=True).unsqueeze(-1).expand(
                        probs.shape[0], self.n_quantiles, 1)) \
                        .transpose(1, 2),
                    T.argmax(probs.mean(dim=1), dim=1)
                )

            def _estimate_action(self, state, state_2):
                x = self.net(state)

                # return shape (batch, quantile, action)
                return x.view(state.shape[0], self.n_quantiles, self.n_actions)

        agent = DQNAgent(
            lambda: Net(30*10, n_actions=env.strategy.action_space.n),
            exit_criteria=lambda reward, cnt: reward > 1 or cnt > 10,
            objective=QuantileHuberLoss(),
            batch_size=13
        )

        agent.fit(env)