from typing import Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from pandas_ml_quant_rl.model.policy.pytorch.torch_network import PolicyNetwork
from pandas_ml_quant_rl.model.policy import Policy
from pandas_ml_quant_rl.model.buffer import ListBuffer
from pandas_ml_quant_rl.model.policy.pytorch.torch_utils import grow_same_ndim


class DQN_Policy(Policy):

    def __init__(self,
                 network: Callable[[], PolicyNetwork],
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32,
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
        self.device = self.network.device
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.nr_episodes_update_target = nr_episodes_update_target
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer = optimizer(self.network.parameters())
        self.objective = objective
        self.verbose = verbose

    def train(self):
        self.network.train()
        self.target_network.eval()
        return super().train()

    def eval(self):
        self.network.eval()
        self.target_network.eval()
        return super().eval()

    def reset(self):
        self._copy_weights()
        self.replay_buffer = ListBuffer(["state", "action", "reward", "state prime", "done"],
                                        max_size=self.replay_buffer_size)
        self.copy_target_cnt = 0
        self.nr_of_episodes = 0
        self.nr_of_batches = 0

    def choose_action(self, env, state, render_on_axis=None):
        with T.no_grad():
            action = self.network(state, render_axis=render_on_axis)[1].cpu().detach().item()

        if self.is_learning_mode:
            # decay epsilon after each batch of episodes
            epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-1. * self.nr_of_batches / self.epsilon_decay)
            if np.random.random() < epsilon:
                return env.sample_action()

        return action

    def learn(self, state, action, reward, new_state, done, info):
        self.replay_buffer.append_args(state, action, reward, new_state, done)

        if len(self.replay_buffer) >= self.batch_size:
            # after we have reached a buffer size of batch size this is equivalent to the number of episodes
            self.nr_of_batches += 1

            # sample experiences from the replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # get the current Q values from the network
            current_q_values = self.network(states, actions)

            # get the target Q values for the "label"
            target_q_values = self._calc_target_q_values(next_states, rewards, dones, self.device)

            # back propagate and clamp overshooting gradients
            loss = self._backprop(current_q_values, target_q_values)
        if done:
            self.nr_of_episodes += 1

            # after n episodes we update our target network or perform a soft update
            if self.nr_episodes_update_target < 0:
                self._soft_update_weights()
            elif self.nr_of_episodes % self.nr_episodes_update_target == 0:
                self._copy_weights()
                self.copy_target_cnt += 1

    def _calc_target_q_values(self, next_states, rewards, dones, device):
        next_state_values = self.target_network(next_states)[0].detach()
        rewards = T.FloatTensor(rewards).to(device)
        dones = T.FloatTensor(dones).to(device)

        next_state_values, rewards, dones = grow_same_ndim(next_state_values, rewards, dones)
        target_q_values = rewards + (self.gamma * next_state_values * (1 - dones))
        return target_q_values

    def _backprop(self, current_q_values, target_q_values):
        input, target = grow_same_ndim(current_q_values, target_q_values)
        loss = self.objective(input, target)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.detach().cpu().item()

    def _copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _soft_update_weights(self):
        local_model = self.network
        target_model = self.target_network.eval()
        TAU = self.nr_episodes_update_target

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU)*target_param.data)

