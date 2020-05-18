from typing import Tuple, Union

import torch as t
import torch.nn as nn
from torch.nn import init


class Time2Vec(nn.Module):
    """
    source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, in_dim: Union[Tuple[int, int], int], out_dim: int):
        super().__init__()
        if not isinstance(in_dim, tuple):
            in_dim = (in_dim, 1)

        self.output_dim = out_dim
        self.w = nn.Parameter(t.Tensor(1, 1))
        self.b = nn.Parameter(t.Tensor(in_dim[0], 1))
        self.W = nn.Parameter(t.Tensor(out_dim * in_dim[1], out_dim * in_dim[1]))
        self.B = nn.Parameter(t.Tensor(in_dim[0], out_dim * in_dim[1]))

    def reset_parameters(self):
        init.uniform_(self.W, 1e-6, 1)
        init.uniform_(self.B, 1e-6, 1)
        init.uniform_(self.w, 1e-6, 1)
        init.uniform_(self.b, 1e-6, 1)

    def forward(self, x):
        if x.ndim < 3:
            x = x.view(*x.shape, 1)

        original = self.w * x + self.b
        x = t.repeat_interleave(x, self.output_dim, dim=-1)
        sin_trans = t.sin((x @ self.W) + self.B)
        return t.cat([sin_trans, original], -1)

