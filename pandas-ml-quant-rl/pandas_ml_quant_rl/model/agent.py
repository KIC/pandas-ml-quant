from typing import Callable, Iterable

from .environment import Environment
from .policy import Policy


class Agent(object):

    def __init__(self, ema_factor=20, render_nr_actions=0, render_env=True, render_policy=True, figsize=(24, 8)):
        self.render_nr_actions = render_nr_actions
        self.ema_factor = 1 / (ema_factor + 1)
        self.render_env = render_env
        self.render_policy = render_policy

        nr_of_subplots = render_env + render_policy
        if nr_of_subplots > 0:
            import matplotlib.pyplot as plt

            if isinstance(nr_of_subplots, tuple):
                fig, ax = plt.subplots(*nr_of_subplots, figsize=figsize)
            else:
                fig, ax = plt.subplots(1, nr_of_subplots, figsize=figsize)

            self.fig = fig
            self.ax = ax if isinstance(ax, Iterable) else [ax]
        else:
            self.fig = None
            self.ax = (None, None)

    def learn_to_play(self, env: Environment, policy: Policy, exit_criteria: Callable[[float, int], bool]):
        policy.reset()
        return self._play_util(env.as_train(), policy.train(), exit_criteria)

    def play_one_episode(self, env: Environment, policy: Policy):
        episode_reward = self._play_episode(env.as_predict(), policy.eval())
        if self.fig is not None:
            import matplotlib.pyplot as plt
            self.fig.suptitle(f"reward {episode_reward}")
            self.fig.canvas.draw()
            plt.close(self.fig)

        return episode_reward

    def _play_util(self, env: Environment, policy: Policy, exit_criteria: Callable[[float, int], bool]):
        total_reward_ma = None
        total_reward = 0
        episodes = 0

        while True:
            episode_reward = self._play_episode(env, policy)
            total_reward += episode_reward
            episodes += 1

            if total_reward_ma is not None:
                total_reward_ma = episode_reward * self.ema_factor + (total_reward_ma * (1 - self.ema_factor))
            else:
                total_reward_ma = episode_reward

            if self.fig is not None:
                self.fig.suptitle(f"reward {episode_reward} / mean {total_reward_ma:.2f}")
                self.fig.canvas.draw()

            if exit_criteria(total_reward_ma, episodes):
                print(f"Solved {episode_reward} (mean {total_reward_ma}) in {episodes} episodes")
                break

        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)

        return total_reward_ma

    def _play_episode(self, env: Environment, policy: Policy):
        # initialize our episode
        state = env.reset()
        episode_reward = 0
        action_counter = 0
        done = False

        while not done:
            # we render the environment and the policy
            do_render = self.render_nr_actions > 0 and action_counter % self.render_nr_actions == 0

            # we use the policy to provide an action
            action = policy.choose_action(env, state, self.ax[-1] if do_render else None)
            action_counter += 1

            # we execute the action in the environment
            new_state, reward, done, info = env.execute_action(action, self.ax[0] if do_render else None)
            episode_reward += reward

            # we trigger a learning function on the policy
            policy.log_experience(state, action, reward, new_state, done, info)
            state = new_state

            # if render draw canvas
            if do_render: self.fig.canvas.draw()

        # render last frame and return reward
        if self.fig is not None:
            env.render(self.ax[0])
            policy.choose_action(env, state, self.ax[-1])

        return episode_reward

# TODO add tensorboard logging
# todo add log chart for action space entropy
#         self.tensorboard_writer = EasySummaryWriter(self.__class__.__name__, **kwargs)
#    def log_to_tensorboard(self, cumulative_reward, nr_of_steps=None, action_hist=None, scalars=None):
#        # log rewards
#        if self._cumulative_reward_ma is None:
#            self._cumulative_reward_ma = cumulative_reward
#        else:
#            self._cumulative_reward_ma = cumulative_reward * (2 / 21) + (self._cumulative_reward_ma * (1 - 2 / 21))
#
#        self.tensorboard_writer.add_scalars(
#            "reward",
#            {"reward": cumulative_reward, "ma": self._cumulative_reward_ma},
#            self._episodes
#        )
#
#        # log nuber of steps per episode
#        if nr_of_steps is not None:
#            self.tensorboard_writer.add_scalar("nr of steps", nr_of_steps, self._episodes)
#
#        # log actions
#        if action_hist is not None:
#            action, counts = np.unique(action_hist, return_counts=True)
#            self.tensorboard_writer.add_scalars(
#                "action counts",
#                {str(a): c for a, c in zip(action, counts)},
#                self._episodes
#            )
#
#        # log epsilon
#        if scalars is not None:
#            for item, number in scalars.items():
#                self.tensorboard_writer.add_scalar(item, number, self._episodes)
#
#        # flush data and increase steps
#        self.tensorboard_writer.flush()
#        self._episodes += 1
#