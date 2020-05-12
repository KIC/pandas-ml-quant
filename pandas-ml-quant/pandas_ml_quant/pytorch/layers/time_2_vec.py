import torch as t
import torch.nn as nn
from torch.nn import init


class Time2Vec(nn.Module):
    """
    source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.W = nn.Parameter(t.Tensor(output_dim, output_dim))
        self.B = nn.Parameter(t.Tensor(input_dim, output_dim))
        self.w = nn.Parameter(t.Tensor(1, 1))
        self.b = nn.Parameter(t.Tensor(input_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.W, 0, 1)
        init.uniform_(self.B, 0, 1)
        init.uniform_(self.w, 0, 1)

    def forward(self, x):
        original = self.w * x + self.b
        x = t.repeat_interleave(x, self.output_dim, dim=-1)
        sin_trans = t.sin(t.dot(x, self.W) + self.B)
        return t.cat([sin_trans, original], -1)

