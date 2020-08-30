from typing import Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .abstract_agent import Agent
from .pytorch.abstract_network import PolicyNetwork
from .pytorch.torch_utils import grow_same_ndim
from pandas_ml_quant_rl.environments import Environment
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

    def fit(self, env: Environment, render_nr_actions=0, render_env=True, render_policy=True, figsize=(20, 8)) -> 'ReinforceAgent':
        fig, ax = Agent.create_plot(render_env + render_policy, figsize) if render_nr_actions else (None, None)
        self._copy_weights()
        net = self.network.train()
        device = net.device
        replay_buffer = ListBuffer(["state", "action", "reward", "state prime", "done"], max_size=self.replay_buffer_size)
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
                new_episode = True
                episode_action_counter = 0
                episode_reward = 0
                done = False

                while not done:
                    # rendering
                    if (render_nr_actions < 0 and new_episode) or (render_nr_actions > 0 and episode_action_counter % render_nr_actions == 0):
                        fig.suptitle(f"mean reward: {mean_episode_reward}")

                        if render_env:
                            env.render(ax[0])
                        if render_policy:
                            with T.no_grad():
                                net(state, render_axis=ax[-1])

                        fig.canvas.draw()

                    # sample and execute action
                    action = self.pick_action(env, state, epsilon)
                    new_state, reward, done, info = env.step(action)
                    new_episode = False

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
                        target_q_values = self._calc_target_q_values(next_states, rewards, dones, device)

                        # back propagate and clamp overshooting gradients
                        self._backprop(current_q_values, target_q_values)

                        # decay epsilon after each batch of episodes
                        epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-1. * nr_of_batches / self.epsilon_decay)

                    if done:
                        nr_of_episodes += 1

                        # calculate the mean reward -> TODO should be part of super
                        if mean_episode_reward is not None:
                            mean_episode_reward = episode_reward * ema_factor + (mean_episode_reward * (1 - ema_factor))
                        else:
                            mean_episode_reward = episode_reward

                        # after n episodes we update our target network or perform a soft update
                        if self.nr_episodes_update_target < 0:
                            self._soft_update_weights()
                        elif self._episodes % self.nr_episodes_update_target == 0:
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

        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

        self.network.eval()
        return self

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

    def _copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _soft_update_weights(self):
        local_model = self.network
        target_model = self.target_network.eval()
        TAU = self.nr_episodes_update_target

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU)*target_param.data)

    def pick_action(self, env, state, epsilon=None, probs=None):
        if epsilon is not None and np.random.random() < epsilon:
            action = env.sample_action(probs)
        else:
            with T.no_grad():
                action = self.network(state)[1].cpu().detach().item()

        # TODO do the rendering here! -> however rendering should be part of super
        #  if (render_nr_actions < 0 and new_episode) or (
        #          render_nr_actions > 0 and episode_action_counter % render_nr_actions == 0):
        #      fig.suptitle(f"mean reward: {mean_episode_reward}")
        #
        #      if render_env:
        #          env.render(ax[0])
        #      if render_policy:
        #          with T.no_grad():
        #              net(state, render_axis=ax[-1])
        #
        #      fig.canvas.draw()
        #
        return action