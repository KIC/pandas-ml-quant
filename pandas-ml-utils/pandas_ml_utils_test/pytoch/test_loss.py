from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_utils import PytorchModel, FeaturesAndLabels
from pandas_ml_utils import pd, np
from pandas_ml_utils.ml.data.splitting import NaiveSplitter
from pandas_ml_utils.pytorch.loss import MultiObjectiveLoss


class TestLoss(TestCase):

    def test_multi_objective_loss(self):
        df = pd.DataFrame(np.array([
            # train
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            # test
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]), columns=["f1", "f2", "l"])

        class XorModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.x1 = nn.Linear(2, 1)
                self.s1 = nn.Sigmoid()
                self.x2 = nn.Linear(2, 1)
                self.s2 = nn.Sigmoid()
                self.s = nn.Softmax()

            def forward(self, x):
                if self.training:
                    return self.s1(self.x1(x)), self.s2(self.x2(x))
                else:
                    return self.s1(self.x1(x))

        fit = df.model.fit(
            PytorchModel(
                FeaturesAndLabels(["f1", "f2"], ["l"]),
                XorModule,
                lambda: MultiObjectiveLoss((1, nn.MSELoss(reduction='none')),
                                           (1, nn.L1Loss(reduction='none')),
                                           on_epoch=lambda criterion, epoch: criterion.update_weights((0, 1.1))),
                Adam
            ),
            NaiveSplitter(0.5)
        )

        print(fit.test_summary.df)


