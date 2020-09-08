import torch as T
import torch.nn as nn
import torch.functional as F


class QuantileHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantile_tau = None

    def huber_loss(self, x, k=1.0):
        return T.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    def forward(self, input, target):
        assert input.ndim == 3, "expected shape (batch, quantile, action)"
        if self.quantile_tau is None:
            batch, quantile, action = input.shape
            self.quantile_tau = T.FloatTensor([i / quantile for i in range(1, quantile + 1)]).to(input.device)

        td_error = target - input
        huber_loss = self.huber_loss(td_error)
        quantile = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss / 1.0
        loss = quantile.sum(dim=1).mean(dim=1)
        loss = loss.mean()
        return loss


