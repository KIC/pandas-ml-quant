from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from pandas_ml_common import Typing
from pandas_ml_utils import Summary, Model, call_callable_dynamic_args
from pandas_ml_utils.constants import *


class PricePredictionSummary(Summary):

    @staticmethod
    def with_reconstructor(
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
            prediction_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = None):
        return lambda df, model, **kwargs: PricePredictionSummary(df, model, label_reconstruction, prediction_reconstruction, **kwargs)


    def __init__(
            self,
            df: Typing.PatchedDataFrame,
            model: Model,
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
            prediction_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
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
        self.label_reconstruction = label_reconstruction
        self.prediction_reconstruction = prediction_reconstruction if prediction_reconstruction is not None else label_reconstruction
        self.dfl = call_callable_dynamic_args(self.label_reconstruction, y=df[LABEL_COLUMN_NAME], df=df)
        self.dfp = call_callable_dynamic_args(self.prediction_reconstruction, y=df[PREDICTION_COLUMN_NAME], df=df).add_suffix(" (Predicted)")

    def plot_prediction(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        self.dfl.plot(ax=ax)
        self.dfp.plot(ax=ax)

        return fig

    def calc_scores(self, *args, **kwargs):
        dflp = self.dfl.pct_change().dropna()
        dfpp = self.dfp.pct_change().dropna()

        idx = dflp.index.intersection(dfpp.index)
        dflp = dflp.loc[idx]
        dfpp = dfpp.loc[idx]

        # todo if multi level index then calculate scores for each top level row
        direction_correct_ratio = ((dflp.values > 0) & (dfpp.values > 0)).sum() / len(dfpp)
        corr, _ = pearsonr(dflp.values.flatten(), dfpp.values.flatten())
        mse = mean_squared_error(dflp, dfpp)
        r2 = r2_score(dflp, dfpp)

        return pd.DataFrame({
            "mse": [mse],
            "Direction Correct Ratio": [direction_correct_ratio],
            "Correlation": [corr],
            "r^2": [r2],
        })
