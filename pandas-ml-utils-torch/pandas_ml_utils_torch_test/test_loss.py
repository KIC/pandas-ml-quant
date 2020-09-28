from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils_torch import PytorchModel
from pandas_ml_utils import pd, np
from pandas_ml_common.sampling import naive_splitter
from pandas_ml_utils_torch.loss import MultiObjectiveLoss, RegularizedLoss
from pandas_ml_utils_torch.layers import RegularizedLayer


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
            naive_splitter(0.5)
        )

        print(fit.test_summary.df)

    def test_regularized_loss(self):

        class TestModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    RegularizedLayer(nn.Linear(10, 2)),
                    nn.ReLU()
                )

            def forward(self, x):
                return self.net(x)

        m = TestModel()
        RegularizedLoss(m.parameters(), nn.MSELoss(reduction='none'))

        # FIXME allow PytorchModel loss with parameters
