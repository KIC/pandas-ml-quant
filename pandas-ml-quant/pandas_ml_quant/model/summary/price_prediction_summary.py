from typing import List, Callable

import pandas as pd
import numpy as np

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
            layout=[[0]],
            **kwargs
        )
        self.label_reconstruction = label_reconstruction
        self.prediction_reconstruction = prediction_reconstruction if prediction_reconstruction is not None else label_reconstruction

    def plot_prediction(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        df = self.df
        dfl = call_callable_dynamic_args(self.label_reconstruction, y=df[LABEL_COLUMN_NAME], df=df)
        dfp = call_callable_dynamic_args(self.prediction_reconstruction, y=df[PREDICTION_COLUMN_NAME], df=df)

        dfl.plot(ax=ax)
        dfp.plot(ax=ax)

        return fig