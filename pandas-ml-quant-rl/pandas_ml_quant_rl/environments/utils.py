import gym
import numpy as np

class SpaceUtils(object):

    @staticmethod
    def unbounded_box(shape):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)

    @staticmethod
    def unbounded_tuple_boxes(*shapes):
        ts = tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape) for shape in shapes])
        return gym.spaces.Tuple(ts)
