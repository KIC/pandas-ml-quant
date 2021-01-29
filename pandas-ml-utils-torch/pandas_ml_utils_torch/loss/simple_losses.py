from typing import Tuple, Callable

import torch as t
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class MultiObjectiveLoss(nn.Module):

    def __init__(self,
                 *criterion: Tuple[int, _Loss],
                 on_epoch: Callable[['MultiObjectiveLoss', t.Tensor], t.Tensor] = None,
                 reduction='mean',
                 max_weight=1,
                 min_weight=0,
                 ):
        super().__init__()
        self.weights = [c[0] for c in criterion]
        self.losses = [c[1] for c in criterion]
        self.on_epoch = on_epoch
        self.reduction = reduction
        self.max_weight = max_weight
        self.min_weight = min_weight

    def forward(self, y_hat, y):
        # calculate individual losses
        losses = [self.losses[i](y_hat[i], y) for i in range(len(self.losses))]

        # eventually reduce dimension to (batch, ) and apply weight on each sample of the bach
        losses = [(loss.sum(dim=1) if loss.ndim > 1 else loss) * self.weights[i] for i, loss in enumerate(losses)]

        # stack losses and return a sum per batch
        if self.reduction == 'sum':
            s = losses[0].sum()
            for i in range(1, len(losses)):
                s += losses[i].sum()

            return s
        elif self.reduction == 'mean':
            s = losses[0].mean()
            for i in range(1, len(losses)):
                s += losses[i].mean()

            return s
        else:
            return t.stack(losses, dim=1).sum(dim=1)

    def update_weights(self, *factors: Tuple[int, float]):
        for factor in factors:
            self.weights[factor[0]] = max(min(self.weights[factor[0]] * factor[1], self.max_weight), self.min_weight)

    def callback(self, epoch):
        if self.on_epoch is not None:
            self.on_epoch(self, epoch)


class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction='none'):
        super().__init__()
        self.cel = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_pred.shape)

        if y_true.ndim > 2:
            if y_true.shape[1] > 1:
                return t.stack([self.forward(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[1])], dim=1)
            else:
                return self.cel(y_pred[:, 0], y_true[:, 0].argmax(dim=-1))
        else:
            return self.cel(y_pred, y_true.argmax(dim=-1))
