from ..environments.abstract_environment import Environment


class Agent(object):

    @staticmethod
    def load(filename) -> 'Agent':
        # FIXME implement load method
        pass

    def __init__(self):
        pass

    def save(self):
        # FIXME implement save method
        pass

    def fit(self, env: Environment) -> 'Agent':
        raise NotImplementedError

    def best_action(self, state):
        raise NotImplementedError

