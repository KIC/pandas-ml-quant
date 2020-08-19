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
            return self.forward([x])

        return self.estimate(*self._unzip_to_tensor(x))

    def _unzip_to_tensor(self, rows):
        if len(rows) < 0:
            return T.FloatTensor(rows).to(self.device)

        columns = []
        for j in range(len(rows[0])):
            if isinstance(rows[0][j], tuple):
                columns.append(self._unzip_to_tensor([row[j] for row in rows]))
            else:
                columns.append(T.FloatTensor(np.vstack([row[j] for row in rows])).to(self.device))

        return tuple(columns)

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
