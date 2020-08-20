from ..environments.abstract_environment import Environment
from ...wrapper.tensorboardX import EasySummaryWriter


class Agent(object):

    @staticmethod
    def load(filename) -> 'Agent':
        # FIXME implement load method
        pass

    def __init__(self, **kwargs):
        self.tensorboard_writer = EasySummaryWriter(self.__class__.__name__, **kwargs)
        self._cumulative_reward_ma = 0
        self._eposides = 0

    def log_episode_reward(self, cumulative_reward):
        # calculate moving average
        self._cumulative_reward_ma = cumulative_reward * (2 / 21) + (self._cumulative_reward_ma * (1 - 2 / 21))
        self.tensorboard_writer.add_scalars(
            "reward",
            {"reward": cumulative_reward, "ma": self._cumulative_reward_ma},
            self._eposides
        )

        self._eposides += 1

    def save(self):
        # FIXME implement save method
        pass

    def fit(self, env: Environment) -> 'Agent':
        raise NotImplementedError

    def best_action(self, state):
        raise NotImplementedError

