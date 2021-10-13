from typing import Dict
from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import Adam

from pandas_ml_common.sampling import naive_splitter
from pandas_ml_common.utils.column_lagging_utils import lag_columns
from pandas_ml_utils import FeaturesLabels, FittingParameter, FittableModel, AutoEncoderModel
from pandas_ml_utils import pd, np
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils_torch import PytorchModelProvider, PytorchNN
from pandas_ml_utils_torch import lossfunction
from .config import TEST_DF


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
            FittableModel(
                PytorchModelProvider(
                    XorModule,
                    lambda: lossfunction.MultiObjectiveLoss(
                        (1, nn.MSELoss(reduction='none')),
                        (1, nn.L1Loss(reduction='none')),
                        on_epoch=lambda criterion, epoch: criterion.update_weights((0, 1.1))),
                    Adam
                ),
                FeaturesLabels(features=["f1", "f2"], labels=["l"]),
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
            FittableModel(
                PytorchModelProvider(
                    TestModel,
                    nn.MSELoss,
                    Adam
                ),
                FeaturesLabels(features=["f"], labels=["l"]),
            ),
            FittingParameter(epochs=1000, splitter=naive_splitter(0.5))
        )

        print(fit.model._cross_validation_models[0].net.net[2].weight.detach().numpy())
        print(fit.model._cross_validation_models[0].net.net[2].weight.norm().detach().item())
        self.assertLess(fit.model._cross_validation_models[0].net.net[2].weight.norm().detach().item(), 0.1)

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
            from pandas_ml_utils import FeaturesLabels
            from pandas_ml_utils_torch import PytorchNN, PytorchModelProvider
            from pandas_ml_utils_torch import lossfunction
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
                FittableModel(
                    PytorchModelProvider(
                        Net,
                        lossfunction.HeteroscedasticityLoss,
                        Adam
                    ),
                    FeaturesLabels(["x"], labels=["y"]),
                ),
                FittingParameter(batch_size=128, epochs=10, splitter=duplicate_data())
            )

        print(df.model.predict(fit.model)[PREDICTION_COLUMN_NAME])
        self.assertEqual((420, 1), df.model.predict(fit.model)[PREDICTION_COLUMN_NAME].values.shape)
        self.assertEqual((420, 2), df.model.predict(fit.model)[PREDICTION_COLUMN_NAME].ML.values.shape)

    def test_soft_dtw_loss(self):
        df = TEST_DF[["Close"]][-21:].copy()

        class LstmAutoEncoder(PytorchNN):
            def __init__(self):
                super().__init__()
                self.input_size = 1
                self.seq_size=10
                self.hidden_size = 2
                self.num_layers = 1
                self.num_directions = 1

                self._encoder =\
                    nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True)

                self._decoder =\
                    nn.RNN(input_size=self.hidden_size, hidden_size=self.input_size, num_layers=self.num_layers,
                           batch_first=True)

            def forward_training(self, x):
                # make sure to treat single elements as batches
                x = x.view(-1, self.seq_size, self.input_size)
                batch_size = len(x)

                hidden_encoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
                hidden_decoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))

                x, _ = self._encoder(x, hidden_encoder)
                x = t.repeat_interleave(x[:,-2:-1], x.shape[1], dim=1)
                x, hidden = self._decoder(x, hidden_decoder)
                return x.squeeze()

            def encode(self, x):
                x = x.reshape(-1, self.seq_size, self.input_size)
                batch_size = len(x)

                with t.no_grad():
                    hidden = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

                    # return last element of sequence
                    return self._encoder(x, hidden)[0][:,-1]

            def decode(self, x):
                x = x.reshape(-1, self.seq_size, self.hidden_size)
                batch_size = len(x)

                with t.no_grad():
                    hidden = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))
                    return self._decoder(x.float(), hidden)[0]

        with df.model() as m:
            fit = m.fit(
                AutoEncoderModel(
                    PytorchModelProvider(
                        LstmAutoEncoder(),
                        lossfunction.SoftDTW,
                        Adam,
                    ),
                    FeaturesLabels(features=df.columns.to_list(),
                                   features_postprocessor=lambda df: lag_columns(df, 10).dropna(),
                                   labels=["condensed-a", "condensed-b"]),
                ),
                FittingParameter(epochs=100)
            )
            print(fit.test_summary.df)

            encoded = df.model.predict(fit.model.as_encoder())
            print(encoded)

