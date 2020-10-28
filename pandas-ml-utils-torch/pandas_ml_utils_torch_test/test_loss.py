from unittest import TestCase

import torch.nn as nn
from torch.optim import Adam

from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils_torch import PytorchModel, PytorchNN
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

        class XorModule(PytorchNN):

            def __init__(self):
                super().__init__()
                self.x1 = nn.Linear(2, 1)
                self.s1 = nn.Sigmoid()
                self.x2 = nn.Linear(2, 1)
                self.s2 = nn.Sigmoid()
                self.s = nn.Softmax()

            def forward_training(self, x):
                return self.s1(self.x1(x)), self.s2(self.x2(x))

            def forward_predict(self, x):
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
        df = pd.DataFrame({"f": np.random.random(20), "l": np.random.random(20)})

        class TestModel(PytorchNN):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 10),
                    nn.ReLU(),
                    RegularizedLayer(nn.Linear(10, 2)),
                    nn.ReLU(),
                    nn.Linear(2, 1),
                    nn.Sigmoid()
                )

            def forward_training(self, x):
                return self.net(x)

        fit = df.model.fit(
            PytorchModel(
                FeaturesAndLabels(["f"], ["l"]),
                TestModel,
                lambda params: RegularizedLoss(params, nn.MSELoss(reduction='none')),
                Adam
            ),
            naive_splitter(0.5)
        )

        # assert no exception is thrown ;-)
