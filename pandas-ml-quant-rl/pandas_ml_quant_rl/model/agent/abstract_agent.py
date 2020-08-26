from typing import List

import numpy as np

from ..environments.abstract_environment import Environment
from ...wrapper.tensorboardX import EasySummaryWriter


class Agent(object):

    @staticmethod
    def load(filename) -> 'Agent':
        # FIXME implement load method
        pass

    def __init__(self, **kwargs):
        self.tensorboard_writer = EasySummaryWriter(self.__class__.__name__, **kwargs)
        self._cumulative_reward_ma = None
        self._episodes = 0

    def log_to_tensorboard(self, cumulative_reward, nr_of_steps=None, action_hist=None, scalars=None):
        # log rewards
        if self._cumulative_reward_ma is None:
            self._cumulative_reward_ma = cumulative_reward
        else:
            self._cumulative_reward_ma = cumulative_reward * (2 / 21) + (self._cumulative_reward_ma * (1 - 2 / 21))

        self.tensorboard_writer.add_scalars(
            "reward",
            {"reward": cumulative_reward, "ma": self._cumulative_reward_ma},
            self._episodes
        )

        # log nuber of steps per episode
        if nr_of_steps is not None:
            self.tensorboard_writer.add_scalar("nr of steps", nr_of_steps, self._episodes)

        # log actions
        if action_hist is not None:
            action, counts = np.unique(action_hist, return_counts=True)
            self.tensorboard_writer.add_scalars(
                "action counts",
                {str(a): c for a, c in zip(action, counts)},
                self._episodes
            )

        # log epsilon
        if scalars is not None:
            for item, number in scalars.items():
                self.tensorboard_writer.add_scalar(item, number, self._episodes)

        # flush data and increase steps
        self.tensorboard_writer.flush()
        self._episodes += 1

    def save(self):
        # FIXME implement save method
        pass

    def fit(self, env: Environment) -> 'Agent':
        raise NotImplementedError

    def play_episode(self, env: Environment, render=True) -> List[float]:
        """
        let the agent play one episode from what he has learned
        :param env: the environment the agent should play
        :param render: whether to render each action or not
        :return: the rewards collected playing this episode
        """
        net = self.network.eval()
        state = env.reset()
        rewards = []
        done = False

        while not done:
            action = self.best_action(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if render:
                env.render()

        return rewards

    def best_action(self, state):
        raise NotImplementedError


