from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
import torch as T
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Union[List, Tuple[np.ndarray]]):
        if isinstance(x, tuple) and len(x) == 2:
            features_state, strategy_state = x
            # convert tuple space to tensors and forward to the estimate function
            tensors = [T.FloatTensor(_x).to(self.device) for _x in x]
        else:
            # we assume we have a list of state tuples and we need to unpack them
            tensors = [T.FloatTensor(np.vstack(_x)).to(self.device) for _x in zip(*x)]
        return self.estimate(*tensors)

    @property
    @lru_cache(1)
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})

        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))

        return next(iter(devices))

    def estimate(self, *tensors: T.Tensor):
        raise NotImplementedError
