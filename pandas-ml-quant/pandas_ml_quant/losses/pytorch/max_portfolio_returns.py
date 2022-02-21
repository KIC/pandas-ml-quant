import torch as T
from torch import nn
from pandas_ml_utils_torch.modules.loss._loss_utils import reduce


class MaximizePortfolioReturns(nn.Module):

    def __init__(self, sparcity_pow=2, sparcity_weight=0, reduction='none'):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.sparcity = sparcity_pow, sparcity_weight
        self.reduction = reduction

    def forward(self, predicted_weights, labeled_asset_returns):
        portfolio_return = T.sum(labeled_asset_returns * predicted_weights, dim=-1)
        sparcity_penalization = T.sum(predicted_weights ** self.sparcity[0], dim=-1) * self.sparcity[1]

        loss = -portfolio_return + sparcity_penalization
        return reduce(loss, self.reduction, loss.view(predicted_weights.shape[0]))