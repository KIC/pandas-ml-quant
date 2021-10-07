from typing import Dict
from unittest import TestCase

import torch as T
import torch.nn as nn
from torch.optim import Adam

from pandas_ml_utils import FeaturesAndLabels, FittingParameter
from pandas_ml_utils_torch import PytorchModel, PytorchNN
from pandas_ml_utils import pd, np
from pandas_ml_common.sampling import naive_splitter
from pandas_ml_utils_torch.loss import MultiObjectiveLoss


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
                XorModule,
                FeaturesAndLabels(["f1", "f2"], ["l"]),
                lambda: MultiObjectiveLoss((1, nn.MSELoss(reduction='none')),
                                           (1, nn.L1Loss(reduction='none')),
                                           on_epoch=lambda criterion, epoch: criterion.update_weights((0, 1.1))),
                Adam
            ),
            FittingParameter(splitter=naive_splitter(0.5))
        )

        print(fit.test_summary.df)

    def test_regularized_loss(self):
        df = pd.DataFrame({
            "f": np.sin(np.linspace(0, 12, 40)),
            "l": np.sin(np.linspace(5, 17, 40))
        })

        class TestModel(PytorchNN):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 3),
                    nn.ReLU(),
                    nn.Linear(3, 2),
                    nn.ReLU(),
                    nn.Linear(2, 1),
                    nn.Sigmoid()
                )

            def forward_training(self, x):
                return self.net(x)

            def L2(self) -> Dict[str, float]:
                return {
                    '**/2/**/weight': 99999999999.99
                }

        fit = df.model.fit(
            PytorchModel(
                TestModel,
                FeaturesAndLabels(["f"], ["l"]),
                nn.MSELoss,
                Adam
            ),
            FittingParameter(epochs=1000, splitter=naive_splitter(0.5))
        )

        print(fit.model._current_model.net.net[2].weight.detach().numpy())
        print(fit.model._current_model.net.net[2].weight.norm().detach().item())
        self.assertLess(fit.model._current_model.net.net[2].weight.norm().detach().item(), 0.1)

    def test_probabilistic(self):
        def create_sine_data(n=300):
            np.random.seed(32)
            n = 300
            x = np.linspace(0, 1 * 2 * np.pi, n)
            y1 = 3 * np.sin(x)
            y1 = np.concatenate((np.zeros(60), y1 + np.random.normal(0, 0.15 * np.abs(y1), n), np.zeros(60)))
            x = np.concatenate((np.linspace(-3, 0, 60), np.linspace(0, 3 * 2 * np.pi, n),
                                np.linspace(3 * 2 * np.pi, 3 * 2 * np.pi + 3, 60)))
            y2 = 0.1 * x + 1
            y = y1 + y2
            return x, y

        df = pd.DataFrame(np.array(create_sine_data(300)).T, columns=["x", "y"])
        with df.model() as m:
            from pandas_ml_utils import FeaturesAndLabels
            from pandas_ml_utils_torch import PytorchNN, PytorchModel
            from pandas_ml_utils_torch.loss import HeteroscedasticityLoss
            from pandas_ml_common.sampling.splitter import duplicate_data
            from torch.optim import Adam
            from torch import nn

            class Net(PytorchNN):

                def __init__(self):
                    super().__init__()
                    self.l = nn.Sequential(
                        nn.Linear(1, 20),
                        nn.ReLU(),
                        nn.Linear(20, 50),
                        nn.ReLU(),
                        nn.Linear(50, 20),
                        nn.ReLU(),
                        nn.Linear(20, 2),
                    )

                def forward_training(self, x):
                    return self.l(x)

            fit = m.fit(
                PytorchModel(
                    Net,
                    FeaturesAndLabels(["x"], ["y"]),
                    HeteroscedasticityLoss,
                    Adam,
                    restore_best_weights=True
                ),
                FittingParameter(batch_size=128, epochs=10, splitter=duplicate_data())
            )

    def test_heteroscedasticity_loss_3d(self):
        from pandas_ml_utils_torch.loss import HeteroscedasticityLoss

        y_true = T.ones((10, 1, 3))
        y_pred = T.randn((10, 2, 3))
        critereon = HeteroscedasticityLoss(multi_nominal_reduction=None)

        t3d = critereon(y_pred, y_true)
        t2d = critereon(y_pred[:, :, 0], y_true[:, :, 0])

        print(t3d, '\n', t2d)
        np.testing.assert_array_equal(t2d.numpy(), t3d[:, 0].numpy())
