import numpy as np


def discount_rewards(rewards, gamma=0.99) -> np.ndarray:
    # discount reward
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])

    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


class TotalAndMeanReward(object):

    def __init__(self, ema_factor=20):
        self.ema_factor = ema_factor
        self.mean_episode_reward = None
        self.total_episodes_reward = 0
        self.nr_of_episodes = 0

    def __call__(self, episode_reward):
        self.total_episodes_reward += episode_reward
        self.nr_of_episodes += 1

        if self.mean_episode_reward is not None:
            self.mean_episode_reward = episode_reward * self.ema_factor + (self.mean_episode_reward * (1 - self.ema_factor))
        else:
            self.mean_episode_reward = episode_reward

        return self.mean_episode_reward