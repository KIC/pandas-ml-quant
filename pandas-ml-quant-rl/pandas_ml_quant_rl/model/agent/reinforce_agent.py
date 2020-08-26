import itertools
from collections import namedtuple
from typing import Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .abstract_agent import Agent
from .pytorch.abstract_network import PolicyNetwork
from .pytorch.losses import LogProbLoss
from .utils import discount_rewards
from ..environments.abstract_environment import Environment
from ...buffer.list_buffer import ListBuffer


class ReinforceAgent(Agent):

    def __init__(self,
                 network: PolicyNetwork,
                 exit_criteria: Callable[[float, int], bool] = lambda _, cnt: cnt > 50,
                 batch_size: int = 32,
                 percentile: int = 70,
                 gamma: float = 0.99,
                 optimizer: Callable[[T.tensor], optim.Optimizer] = lambda params: optim.Adam(params=params, lr=0.01),
                 objective=LogProbLoss(),
                 verbose=0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.network = network
        self.exit_criteria = exit_criteria
        self.batch_size = batch_size
        self.percentile = percentile
        self.gamma = gamma
        self.optimizer = optimizer(network.parameters())
        self.objective = objective
        self.verbose = verbose

    def fit(self, env: Environment) -> 'ReinforceAgent':
        # Set up replay buffer
        net = self.network.train()
        device = net.device
        batch_buffer = ListBuffer(["reward", "action", "state"])
        ema_factor = 1 / 21  # 20 episodes weighted average
        mean_episode_reward = None
        episode_counter = 0
        batch_counter = 0
        total_rewards = 0

        try:
            while True:
                # get the initial observed state (start a new game)
                episode_buffer = ListBuffer(["reward", "action", "state"])
                episode_action_counter = 0
                s_0 = env.reset()
                done = False

                while not done:
                    # Get action probabilities and convert to numpy array
                    action_probs = net(s_0)[0].cpu().detach().numpy()
                    # choose action based on probabilities
                    action = env.sample_action(action_probs)
                    # execute action in the environment
                    s_1, reward, done, _ = env.step(action)

                    # keep the trajectory of state, reward and actions
                    episode_buffer.append_args(reward, action, s_0)

                    # update the last state to the current state after we executed the action
                    s_0 = s_1

                    # keep track of number of actions we have executed in this episode
                    episode_action_counter += 1

                    # If done, batch data
                    if done:
                        # store all state, action, reward of each step in the batch buffer
                        batch_buffer.append_args(
                            discount_rewards(episode_buffer["reward"], self.gamma),
                            episode_buffer["action"],
                            episode_buffer["state"]
                        )

                        episode_reward = sum(episode_buffer["reward"])
                        total_rewards += episode_reward
                        episode_counter += 1
                        batch_counter += 1

                        # monitor achieved rewards
                        if mean_episode_reward is not None:
                            mean_episode_reward = episode_reward * ema_factor + (mean_episode_reward * (1 - ema_factor))
                        else:
                            mean_episode_reward = episode_reward

                        # after each episode log some information
                        if self.verbose:
                            print(f"episode_reward: {episode_reward}, nr_of_steps: {episode_action_counter} "
                                  f"actions: {np.unique(batch_buffer['action'][-1], return_counts=True)}")

                        self.log_to_tensorboard(
                            episode_reward,
                            nr_of_steps=episode_action_counter,
                            action_hist=batch_buffer["action"][-1] if len(batch_buffer["action"]) > 0 else None
                        )

                        # If batch is complete, update network
                        if batch_counter >= self.batch_size:
                            # now we need to chain all state, action rewards into one batch
                            states = list(itertools.chain(*batch_buffer["state"]))
                            rewards = list(itertools.chain(*batch_buffer["reward"]))
                            actions = list(itertools.chain(*batch_buffer["action"]))

                            # Calculate loss
                            self.optimizer.zero_grad()
                            loss = self.objective(
                                net(states),
                                T.LongTensor(actions).to(device),
                                T.FloatTensor(rewards).to(device)
                            )

                            # Calculate gradients
                            loss.backward()

                            # Apply gradients
                            self.optimizer.step()

                            # clean up and prepare buffer for next batch
                            batch_buffer = batch_buffer.reset()
                            batch_counter = 0

                # check if we stop training
                if self.exit_criteria(mean_episode_reward, episode_counter):
                    print(f"Solved {mean_episode_reward} in {episode_counter} episodes")
                    break

        except Exception as e:
            # allow interruption
            raise e

        return self

    def best_action(self, state):
        return self.network.eval()(state)[0].cpu().detach().numpy().argmax()



