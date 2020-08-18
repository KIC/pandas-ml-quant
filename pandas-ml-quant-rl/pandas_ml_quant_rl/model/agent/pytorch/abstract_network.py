from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
import torch as T
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Union[List, Tuple[np.ndarray, np.ndarray]]):
        if isinstance(x, tuple) and len(x) == 2:
            features_state, strategy_state = x
            # convert tuple space to tensors and forward to the estimate function
            # FIXME don't rely on two tensors make this a vararg!!
            env_state_tensor, strategy_state_tensor = T.FloatTensor(x[0]), T.FloatTensor(x[1])
        else:
            # we assume we have a list of state tuples and we need to unpack them
            env_state_tensor, strategy_state_tensor = (
                T.FloatTensor(np.vstack([t[0] for t in x])),
                T.FloatTensor(np.vstack([t[1] for t in x]))
            )

        return self.estimate(
            T.FloatTensor(env_state_tensor).to(self.device),
            T.FloatTensor(strategy_state_tensor).to(self.device)
        )

    @property
    @lru_cache(1)
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})

        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))

        return next(iter(devices))

    def estimate(self, feature_state: T.Tensor, portfolio_state: T.Tensor):
        raise NotImplementedError
