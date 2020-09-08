import numpy as np


class Policy(object):

    def __init__(self):
        self.is_learning_mode = True

    def eval(self):
        self.is_learning_mode = False
        return self

    def train(self):
        self.is_learning_mode = True
        return self

    def reset(self):
        pass

    def choose_action(self, env, state, render_on_axis=None):
        pass

    def log_experience(self, state, action, reward, new_state, done, info):
        if self.is_learning_mode:
            self.learn(state, action, reward, new_state, done, info)

    def learn(self, state, action, reward, new_state, done, info):
        pass


