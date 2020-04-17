from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import Adam

from pandas_ml_quant.pytorch.custom_loss import SoftDTW
from pandas_ml_quant_test.config import DF_TEST
from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_utils import AutoEncoderModel, FeaturesAndLabels
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel


class TestCustomLoss(TestCase):

    def test_soft_dtw_loss(self):
        df = DF_TEST[["Close"]][-21:].copy()

        class LstmAutoEncoder(nn.Module):
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

            def forward(self, x):
                # make sure to treat single elements as batches
                x = x.view(-1, self.seq_size, self.input_size)
                batch_size = len(x)

                hidden_encoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
                hidden_decoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))

                x, _ = self._encoder(x, hidden_encoder)
                x = t.repeat_interleave(x[:,-2:-1], x.shape[1], dim=1)
                x, hidden = self._decoder(x, hidden_decoder)
                return x

            def encoder(self, x):
                x = x.reshape(-1, self.seq_size, self.input_size)
                batch_size = len(x)

                with t.no_grad():
                    hidden = nn.Parameter(
                        t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

                    # return last element of sequence
                    return self._encoder(t.from_numpy(x).float(), hidden)[0].numpy()[:,-1]

            def decoder(self, x):
                x = x.reshape(-1, self.seq_size, self.hidden_size)
                batch_size = len(x)

                with t.no_grad():
                    hidden = nn.Parameter(
                        t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))
                    return self._decoder(t.from_numpy(x).float(), hidden)[0].numpy()

        model = AutoEncoderModel(
            PytorchModel(
                PostProcessedFeaturesAndLabels(df.columns.to_list(), [lambda df: df.ta.rnn(10)],
                                               df.columns.to_list(), [lambda df: df.ta.rnn(10)]),
                LstmAutoEncoder,
                SoftDTW,
                Adam
            ),
            ["condensed-a", "condensed-b"],
            lambda m: m.module.encoder,
            lambda m: m.module.decoder,
        )

        fit = df.model.fit(model, epochs=100)
        print(fit.test_summary.df)

        encoded = df.model.predict(fit.model.as_encoder())
        print(encoded)
