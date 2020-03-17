from unittest import TestCase

from sklearn.neural_network import MLPClassifier

import pandas_ml_quant
from pandas_ml_utils import FeaturesAndLabels, SkModel
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor
from pandas_ml_utils.ml.data.sampeling import KFoldBoostRareEvents, KEquallyWeightEvents
from pandas_ml_utils.ml.summary import ClassificationSummary
from test.config import DF_TEST

print(pandas_ml_quant.__version__)


class TestModel(TestCase):

    def test_simple_model(self):
        df = DF_TEST.copy()

        # TODO
        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].q.ta_rsi().q.ta_rnn(280),
                        lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_rnn(280)
                    ],
                    labels=[
                        lambda df: (df["Close"] > df["Open"]).shift(-1),
                    ],
                    min_required_samples=280
                ),
                # kwargs
                forecasting_time_steps=7
            )
        )

        print(fit)

        prediction = df.model.predict(fit.model)
        backtest = df.model.backtest(fit.model)


    def test_hyper_parameter_for_simple_model(self):

        pass

    def test_KFold(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df["Close"].q.ta_trix(),
                            lambda df: df["Close"].q.ta_ppo(),
                            lambda df: df["Close"].q.ta_apo(),
                            lambda df: df["Close"].q.ta_macd(),
                            lambda df: df.q.ta_adx(),
                        ],
                        lambda df: df.q.ta_rnn(range(100))
                    ),
                    labels=[
                        lambda df: df["Close"].q.ta_sma(period=60) \
                            .q.ta_cross(df["Close"].q.ta_sma(period=20)) \
                            .q.ta_rnn([1, 2, 3, 4, 5]) \
                            .abs() \
                            .sum(axis=1) \
                            .shift(-5) \
                            .astype(bool)

                    ],
                    min_required_samples=100
                )
            ),
            test_size=0.4,
            test_validate_split_seed=42,
            cross_validation=(1, KFoldBoostRareEvents(n_splits=5).split)
        )

    def test_debug(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, warm_start=True),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df["Close"].q.ta_macd().ml[['macd.*', 'signal.*']],
                            lambda df: df.q.ta_adx().ml[['+DI', '-DM', '+DM']],
                            lambda df: df["Close"].q.ta_mom(),
                            lambda df: df["Close"].q.ta_apo(),
                            lambda df: df.q.ta_atr(),
                            lambda df: df["Close"].q.ta_trix(),
                        ],
                        lambda df: df.q.ta_rnn(280)
                    ),
                    labels=[
                        lambda df: df["Close"].q.ta_future_bband_quantile().q.ta_one_hot_encode_discrete()

                    ],
                    min_required_samples=280
                ),
                summary_provider=ClassificationSummary,
            ),
            test_size=0.4,
            test_validate_split_seed=42,
            cross_validation=(1, KEquallyWeightEvents(n_splits=3).split),
        )

        print(fit)
        # FIXME test self.asstr(fit)

        prediction = df.model.predict(fit.model, tail=3)

