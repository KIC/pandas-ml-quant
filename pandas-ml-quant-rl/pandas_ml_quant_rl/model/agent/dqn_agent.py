from typing import Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .abstract_agent import Agent
from .pytorch.abstract_network import PolicyNetwork
from ..environments.abstract_environment import Environment
from ...buffer.abstract_buffer import Buffer
from ...buffer.array_buffer import ArrayBuffer


class DQNAgent(Agent):

    def __init__(self,
                 network: Callable[[], PolicyNetwork],
                 exit_criteria: Callable[[float, int], bool] = lambda total_reward, cnt: cnt > 50 or total_reward < -0.2,
                 replay_buffer: Buffer = ArrayBuffer((1000, 4), dtype=object),
                 batch_size: int = 32,
                 percentile: int = 0,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.99,
                 min_epsilon: float = 0.01,
                 nr_episodes_update_target: int = 10,
                 optimizer: Callable[[T.tensor], optim.Optimizer] = lambda params: optim.Adam(params=params, lr=0.01),
                 objective=nn.MSELoss(),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.network = network()
        self.target_network = network()
        self.replay_buffer = replay_buffer
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

    def fit(self, env: Environment) -> 'ReinforceAgent':
        epsilon = self.epsilon
        self._copy_weights()
        ema_factor = 1 / 21  # 20 episodes average
        mean_episode_reward = None
        copy_target_cnt = 0
        nr_of_episodes = 0

        while True:
            state = env.reset()
            episode_action_counter = 0
            episode_reward = 0
            done = False

            while not done:
                experiences = None

                # sample action
                if np.random.random() < epsilon:
                    action = env.sample_action()
                else:
                    with T.no_grad():
                        action = self.network(state).cpu().detach().numpy()[0].argmax()

                new_state, reward, done, info = env.step(action)
                self.replay_buffer.append_row(np.array([state, action, reward, new_state]))
                episode_reward += reward
                state = new_state

                if len(self.replay_buffer) >= self.batch_size:
                    # sample experiences from the replay buffer
                    experiences = self.replay_buffer.sample(self.batch_size)

                    # get the current Q values from the network
                    current_q_values = self.network(experiences[:, 0])\
                        .gather(dim=1, index=T.LongTensor(experiences[:, 1].astype(float)).to(self.network.device).unsqueeze(-1))

                    # get the target Q values for the "label"
                    if not done:
                        target_q_values = self.gamma * self.target_network(experiences[:, 3]).max(dim=1).values
                    else:
                        target_q_values = T.FloatTensor(experiences[:, 2].astype(float)).to(self.network.device)

                    loss = self.objective(current_q_values, target_q_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # decay epsilon
                    epsilon = max(self.min_epsilon, epsilon ** nr_of_episodes)

                if done:
                    if mean_episode_reward is not None:
                        mean_episode_reward = episode_reward * ema_factor + (mean_episode_reward * (1 - ema_factor))
                    else:
                        mean_episode_reward = episode_reward

                    if episode_reward > max_episode_reward: max_episode_reward = episode_reward
                    episode_action_counter += 1

                    if self._episodes % self.nr_episodes_update_target == 0:
                        self._copy_weights()
                        copy_target_cnt += 1

                    self.log_to_tensorboard(
                        episode_reward,
                        episode_action_counter,
                        action_hist=experiences[:, 1] if experiences is not None else None,
                        scalars={
                            "epsilon": epsilon,
                            "target copy": copy_target_cnt
                        }
                    )

                    if self.exit_criteria(mean_episode_reward, nr_of_episodes):
                        print(f"Solved {max_episode_reward} in {nr_of_episodes} episodes")
                        return self

    def _copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
