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
        return self.play_util(env.as_train(), policy.train(), exit_criteria)

    def play_util(self, env: Environment, policy: Policy, exit_criteria: Callable[[float, int], bool]):
        total_reward_ma = None
        total_reward = 0
        episodes = 0

        while True:
            episode_reward = self.play_episode(env, policy)
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

    def play_episode(self, env: Environment, policy: Policy):
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

        # execute None action in env to trigger last rendering frame
        return episode_reward

