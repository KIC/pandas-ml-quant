from collections import namedtuple
from typing import Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .abstract_agent import Agent
from .pytorch.abstract_network import Network
from ..environments.abstract_environment import Environment


class ReinforceAgent(Agent):

    def __init__(self,
                 network: Network,
                 exit_criteria: Callable[[float, int], bool] = lambda _, cnt: cnt > 50,
                 batch_size: int = 32,
                 percentile: int = 70,
                 gamma: float = 0.99,
                 entropy_beta: float = 0.01,
                 optimizer: Callable[[T.tensor], optim.Optimizer] = lambda params: optim.Adam(params=params, lr=0.01),
                 # FIXME add loss function as well so we can i.e. provide KLdivergence as well
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.network = network
        self.exit_criteria = exit_criteria
        self.batch_size = batch_size
        self.percentile = percentile
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.optimizer = optimizer(network.parameters())

    def fit(self, env: Environment):
        Episode = namedtuple('Episode', field_names=['reward', 'observations', 'actions'])
        objective = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)

        # play BATCH_SIZE episodes using random actions and store all trajectories
        def iterate_batches():
            episode_reward = 0.0
            observation_hist = []
            action_hist = []
            batch = []
            i = 0

            # start new game and get initial state
            obs = env.reset()

            while True:
                # use network to get probabilities of  actions
                # then convert the result to numpy and get rid of the batch dimension
                act_probs = softmax(self.network(obs)).data.numpy()[0]

                # decide for a random action (following the probabilities), execute it and collect the reward
                action = env.sample_action(act_probs)
                next_obs, reward, is_done, _ = env.step(action)
                episode_reward += self.gamma ** i * reward

                # remember in which state we were and which action we took
                observation_hist.append(obs)
                action_hist.append(action)

                # we play batch_size games until we die.
                # we collect all EpisodeSteps (state + action) and the rewards (using the Episode tuple) into one
                # sample of a batch. so each sample in the bach potentially has a different length
                # once we have a full batch we 'yield' it. this is done infinitely meaning the training loop needs to
                # decide when to stop
                if is_done:
                    # log reward to tensor board
                    self.log_to_tensorboard(episode_reward, i, action_hist)

                    # store the discounted reward together with all episode state and actions as one sample of a batch
                    # then reset all variables for a new episode
                    batch.append(Episode(reward=episode_reward, observations=observation_hist, actions=action_hist))
                    episode_reward = 0.0
                    observation_hist = []
                    action_hist = []
                    i = 0

                    next_obs = env.reset()

                    # once we have collected batch_size episodes (of various length of actions we could execute)
                    # yield it and reset the batch collection
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

                # move one step forward (eventually we start with a new environment)
                obs = next_obs
                i += 1

        # from a batch of episodes we want to know which were the episodes with the best rewards
        # we can then use this episodes as label. meaning we use successful episodes resp the actions we executed in an
        # environment and try to repeat them.
        def filter_batch(batch):
            # collect all episode rewards of the batch
            rewards = list(map(lambda s: s.reward, batch))
            # sort the rewards and get the element at the percentile index
            reward_bound = np.percentile(rewards, self.percentile)
            # get the mean reward of all rewards
            reward_mean = float(np.mean(rewards))

            # we are only interested in rewards which are better or equal to the percentilest reward
            # collect all such observations and actions we from one batch into a new batch of observations and actions
            train_obs = []; train_act = []
            for example in batch:
                if example.reward >= reward_bound:
                    # collect the observations and actions
                    train_obs.extend(example.observations)
                    train_act.extend(example.actions)

            # return a list of states and actions as well as the reward bound and the mean
            # the later two are only used for logging purposes
            return train_obs, train_act, reward_bound, reward_mean

        # we draw batches form a generator process which plays episodes until one batch is full
        trained_episodes = 0
        for iter_no, sampled_batch in enumerate(iterate_batches()):
            # we then filter those episodes and only those with a reward >= a percentile given as hyper parameter
            obs, acts, reward_bound, reward_mean = filter_batch(sampled_batch)
            trained_episodes += len(acts)

            # check if we have learned enough
            if self.exit_criteria(reward_mean, trained_episodes):
                print(f"Solved! reward: {reward_mean}, episodes: {trained_episodes}")
                break

            # reset optimizer gradients
            self.optimizer.zero_grad()
            # get action probabilities from the policy estimating network
            action_scores_v = self.network(obs)

            # since each batch contains different trajectory of steps and actions we will have different batch sizes for
            # each batch of episodes!
            # NOTE the CrossEntropyLoss expects raw logits as input (raw probabilities for each action)
            # and an integer of the correct action. if the labels would be one hot encoded we would need an argmax!
            loss_v = objective(action_scores_v, T.LongTensor(acts).to(self.network.device))

            # calculate the gradients and update the weights in the network
            loss_v.backward()
            self.optimizer.step()

            """
            TODO use this logic instead !!! 
            # subtract the entropy bonus from the loss function
            prob_v = F.softmax(logits_v, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v

            loss_v.backward()
            optimizer.step()
            """

            # log some information
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f performance=%.2f" % (
                iter_no, loss_v.item(), reward_mean, reward_bound, 0.0))

        return self

    def test(self, env: Environment):
        raise NotImplementedError

    def predict(self, env: Environment):
        raise NotImplementedError


