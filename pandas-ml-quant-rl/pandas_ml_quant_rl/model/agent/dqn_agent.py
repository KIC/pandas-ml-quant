from typing import Callable, List

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .abstract_agent import Agent
from .pytorch.abstract_network import PolicyNetwork
from .pytorch.torch_utils import grow_same_ndim
from ..environments.abstract_environment import Environment
from ...buffer.abstract_buffer import Buffer
from ...buffer.list_buffer import ListBuffer


class DQNAgent(Agent):

    def __init__(self,
                 network: Callable[[], PolicyNetwork],
                 exit_criteria: Callable[[float, int], bool] = lambda total_reward, cnt: cnt > 50 or total_reward < -0.2,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32,
                 percentile: int = 0,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 100,
                 min_epsilon: float = 0.05,
                 nr_episodes_update_target: int = 10,
                 optimizer: Callable[[T.tensor], optim.Optimizer] = lambda params: optim.Adam(params=params, lr=0.01),
                 objective=nn.MSELoss(),
                 verbose=0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.network = network()
        self.target_network = network()
        self.replay_buffer_size = replay_buffer_size
        self.exit_criteria = exit_criteria
        self.batch_size = batch_size
        self.percentile = percentile
        self.nr_episodes_update_target = nr_episodes_update_target
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer = optimizer(self.network.parameters())
        self.objective = objective
        self.verbose = verbose

    def fit(self, env: Environment) -> 'ReinforceAgent':
        self._copy_weights()
        net = self.network.train()
        device = net.device
        replay_buffer =  ListBuffer(["state", "action", "reward", "state prime", "done"], max_size=self.replay_buffer_size)
        epsilon = self.epsilon
        ema_factor = 1 / 21  # 20 episodes weighted average
        mean_episode_reward = None
        copy_target_cnt = 0
        nr_of_episodes = 0
        nr_of_batches = 0
        actions = None

        try:
            while True:
                state = env.reset()
                episode_action_counter = 0
                episode_reward = 0
                done = False

                while not done:
                    # sample action
                    if np.random.random() < epsilon:
                        action = env.sample_action()
                    else:
                        with T.no_grad():
                            action = net(state)[1].cpu().detach().item()

                    new_state, reward, done, info = env.step(action)

                    # we only append non final state to the replay buffer as we would need to filter them out later
                    replay_buffer.append_args(state, action, reward, new_state, done)
                    episode_action_counter += 1
                    episode_reward += reward
                    state = new_state

                    if len(replay_buffer) >= self.batch_size:
                        # after we have reached a buffer size of batch size this is equivalent to the number of episodes
                        nr_of_batches += 1

                        # sample experiences from the replay buffer
                        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)

                        # get the current Q values from the network
                        current_q_values = net(states, actions)

                        # get the target Q values for the "label"
                        next_state_values = self.target_network(next_states)[0].detach()
                        rewards = T.FloatTensor(rewards)
                        dones = T.FloatTensor(dones)

                        next_state_values, rewards, dones = grow_same_ndim(next_state_values, rewards, dones)
                        target_q_values = rewards + (self.gamma * next_state_values * (1 - dones))

                        # back propagate and clamp overshooting gradients
                        loss = self.objective(current_q_values, target_q_values.unsqueeze(1))
                        self.optimizer.zero_grad()
                        loss.backward()

                        for param in net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        self.optimizer.step()

                        # decay epsilon after each batch of episodes
                        epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-1. * nr_of_batches / self.epsilon_decay)

                    if done:
                        nr_of_episodes += 1

                        # calculate the mean reward
                        if mean_episode_reward is not None:
                            mean_episode_reward = episode_reward * ema_factor + (mean_episode_reward * (1 - ema_factor))
                        else:
                            mean_episode_reward = episode_reward

                        # after n episodes we update our target network
                        if self._episodes % self.nr_episodes_update_target == 0:
                            self._copy_weights()
                            copy_target_cnt += 1

                        # and log some information to our tensorboard
                        # after each episode log some information
                        if self.verbose:
                            print(f"episode_reward: {episode_reward}, nr_of_steps: {episode_action_counter} "
                                  f"actions: {np.unique(actions, return_counts=True)}")

                        self.log_to_tensorboard(
                            episode_reward,
                            episode_action_counter,
                            action_hist=actions if actions is not None else None,
                            scalars={
                                "epsilon": epsilon,
                                "target copy": copy_target_cnt
                            }
                        )

                if self.exit_criteria(mean_episode_reward, nr_of_episodes):
                    print(f"Solved {mean_episode_reward} in {nr_of_episodes} episodes")
                    break

        except Exception as e:
            # allow interruption
            raise e

        return self

    def _copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def best_action(self, state):
        return self.network.eval()(state)[1].cpu().detach().item()