from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from pandas_ml_common import Typing
from pandas_ml_utils import Summary, Model, call_callable_dynamic_args
from pandas_ml_utils.constants import *
import scipy.stats as st


class PricePredictionSummary(Summary):

    @staticmethod
    def with_reconstructor(
            label_returns: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y,
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y.ta.cumret(),
            predicted_returns: Callable[[pd.DataFrame], pd.DataFrame] = lambda y_hat: y_hat,
            predicted_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = lambda y_hat: y_hat.ta.cumret(),
            predicted_confidence: Callable[[pd.DataFrame], pd.DataFrame] = lambda y, y_hat: np.sqrt(((y_hat.values - y.values) ** 2).mean()),
            **kwargs):
        return lambda df, model, **kwargs2: PricePredictionSummary(
            df,
            model,
            label_returns,
            label_reconstruction,
            predicted_returns,
            predicted_reconstruction,
            predicted_confidence,
            **{**kwargs, **kwargs2}
        )


    def __init__(
            self,
            df: Typing.PatchedDataFrame,
            model: Model,
            label_returns: Callable[[pd.DataFrame], pd.DataFrame],
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
            predicted_returns: Callable[[pd.DataFrame], pd.DataFrame],
            predicted_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
            predicted_confidence: Callable[[pd.DataFrame], pd.DataFrame],
            confidence: Union[float, Tuple[float, float]] = 0.95,
            figsize=(16, 16),
            **kwargs):
        super().__init__(
            df.sort_index(),
            model,
            self.plot_prediction,
            self.calc_scores,
            layout=[[0],
                    [1]],
            **kwargs
        )
        self.figsize = figsize
        self.label_returns = call_callable_dynamic_args(label_returns, y=df[LABEL_COLUMN_NAME], df=df)
        self.label_reconstruction = call_callable_dynamic_args(label_reconstruction, y=self.label_returns, df=df)
        self.predicted_returns = call_callable_dynamic_args(predicted_returns, y_hat=df[PREDICTION_COLUMN_NAME], df=df)
        self.prediction_reconstruction = call_callable_dynamic_args(predicted_reconstruction, y_hat=self.predicted_returns, df=df)
        self.predicted_confidence = call_callable_dynamic_args(predicted_confidence, y=self.label_returns, y_hat=self.predicted_returns, df=df)
        self.expected_confidence = np.sum(confidence)

        self.confidence = (
            np.abs(st.norm.ppf(confidence[0])),  # i.e 0.025
            np.abs(st.norm.ppf(confidence[1]))   # i.e 0.972
        ) if isinstance(confidence, tuple) else (
            np.abs(st.norm.ppf((1. - confidence) / 2.)),
        ) * 2

        self.lower = (self.predicted_returns - self.predicted_confidence * self.confidence[0]).values.squeeze()
        self.upper = (self.predicted_returns + self.predicted_confidence * self.confidence[1]).values.squeeze()

    def plot_prediction(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        try:
            # comptound
            self.label_reconstruction.dropna().plot(ax=ax[0])
            self.prediction_reconstruction.dropna().plot(ax=ax[0])
            ax[0].legend(['Label', 'Prediction'])

            # returns
            self.label_returns.plot(ax=ax[1])
            self.predicted_returns.plot(ax=ax[1])
            ax[1].fill_between(self.predicted_returns.index, self.lower, self.upper, color='orange', alpha=.25)
            ax[1].legend(['Label', 'Prediction'])
        except Exception as e:
            print(e)

        return fig

    def calc_scores(self, *args, **kwargs):
        dfpp = self.predicted_returns.dropna()
        idx = dfpp.index.intersection(self.label_returns.index)
        dflp = self.label_returns.loc[idx]

        # todo if multi level index then calculate scores for each top level row
        direction_correct_ratio = \
            (((dflp.values.squeeze() > 0) & (dfpp.values.squeeze() > 0)) | ((dflp.values.squeeze() < 0) & (dfpp.values.squeeze() < 0))).sum() / len(dflp)

        corr, _ = pearsonr(dflp.values.flatten(), dfpp.values.flatten())
        mse = mean_squared_error(dflp, dfpp)
        r2 = r2_score(dflp, dfpp)

        # calculate confidence ratio
        tail_events = \
            (self.label_returns.values.squeeze() > self.upper).sum() + \
            (self.label_returns.values.squeeze() < self.lower).sum()

        return pd.DataFrame({
            "mse": [mse],
            "Direction Correct Ratio": [direction_correct_ratio],
            "Correlation": [corr],
            "r^2": [r2],
            "σ": [np.mean(self.predicted_confidence)],
            f"confidence (exp: {self.expected_confidence:.2f})": 1 - tail_events / len(dfpp)
        }).T
