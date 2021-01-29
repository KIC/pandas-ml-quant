from unittest import TestCase

from sklearn.model_selection import KFold
from torch import nn
from torch.optim import SGD

from pandas_ml_common_test.config import TEST_DF
from pandas_ml_utils import FeaturesAndLabels, FittingParameter
from pandas_ml_utils_torch import PytorchNN, PytorchModel
from pandas_ml_utils_torch.merging_cross_folds import take_the_best


class TestCrossFolding(TestCase):

    def test_take_best_model(self):
        df = TEST_DF[["Close"]][-21:].copy()
        with self.assertLogs(level='INFO') as log_ctx:
            with df.model() as m:

                class Net(PytorchNN):

                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(1, 2),
                            nn.Tanh(),
                            nn.Linear(2, 1),
                            nn.Tanh(),
                        )

                    def forward_training(self, x):
                        return self.net(x)

                fit = m.fit(
                    PytorchModel(
                        Net,
                        FeaturesAndLabels(["Close"], [lambda df: df["Close"].shift(-1)]),
                        nn.MSELoss,
                        lambda params: SGD(params, lr=0.01, momentum=0.0),
                        merge_cross_folds=take_the_best
                    ),
                    FittingParameter(cross_validation=KFold(3), fold_epochs=100, epochs=2)
                )


            print('\n'.join(log_ctx.output))
            self.assertIn('INFO:pandas_ml_utils_torch.merging_cross_folds:best fold: 2!', log_ctx.output)

        self.assertGreater(len(fit.test_summary.df), 0)
