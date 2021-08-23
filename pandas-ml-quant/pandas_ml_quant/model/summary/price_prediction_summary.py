import traceback
from typing import Callable, Tuple, Union, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from pandas_ml_common import Typing
from pandas_ml_quant.empirical import ECDF
from pandas_ml_utils import Summary, Model, call_callable_dynamic_args
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.confidence import NormalConfidence


class PricePredictionSummary(Summary):

    @staticmethod
    def with_reconstructor(
            label_returns: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y,
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y.ta.cumret(),
            predicted_returns: Callable[[pd.DataFrame], pd.DataFrame] = lambda y_hat: y_hat,
            predicted_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = lambda y_hat: y_hat.ta.cumret(),
            predicted_std: Callable[[pd.DataFrame], pd.DataFrame] = lambda y, y_hat: np.sqrt(((y_hat.values - y.values) ** 2).mean()),
            **kwargs):
        return lambda df, model, **kwargs2: PricePredictionSummary(
            df,
            model,
            label_returns,
            label_reconstruction,
            predicted_returns,
            predicted_reconstruction,
            predicted_std,
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
            predicted_std: Callable[[pd.DataFrame], pd.DataFrame],
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
        self.prediction_reconstruction = call_callable_dynamic_args(predicted_reconstruction, y_hat=self.predicted_returns, df=df, y=self.label_reconstruction)

        # confidence intervals
        self.expected_confidence = np.sum(confidence)
        self.normal_confidence = NormalConfidence(confidence)

        self.predicted_std = call_callable_dynamic_args(predicted_std, y=self.label_returns, y_hat=self.predicted_returns, df=df)
        if isinstance(self.predicted_std, float):
            self.predicted_std = pd.Series(np.ones(len(self.predicted_returns)) * self.predicted_std, index=self.predicted_returns.index)

        self.lower = pd.concat([self.predicted_returns, self.predicted_std], join='inner', axis=1).apply(self.normal_confidence.lower, axis=1)
        self.upper = pd.concat([self.predicted_returns, self.predicted_std], join='inner', axis=1).apply(self.normal_confidence.upper, axis=1)

    def plot_prediction(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        try:
            # compound
            comp_lower = pd.concat([self.prediction_reconstruction, self.lower + 1], join='inner', axis=1).apply(np.prod, axis=1).dropna()
            comp_upper = pd.concat([self.prediction_reconstruction, self.upper + 1], join='inner', axis=1).apply(np.prod, axis=1).dropna()
            # print(comp_lower.shape, comp_upper.shape)

            self.label_reconstruction.dropna().plot(ax=ax[0])
            self.prediction_reconstruction.dropna().plot(ax=ax[0], color='orange')
            ax[0].fill_between(comp_lower.index, comp_lower.values.squeeze(), comp_upper.values.squeeze(), color='orange', alpha=.25)
            ax[0].legend(['Label', 'Prediction'], loc='upper left')

            # returns
            self.label_returns.plot(ax=ax[1])
            self.predicted_returns.plot(ax=ax[1])
            ax[1].fill_between(self.predicted_returns.index, self.lower, self.upper, color='orange', alpha=.25)
            ax[1].legend(['Label', 'Prediction'], loc='upper left')
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
        left_tail_events_mask = (self.label_returns.values.squeeze() < self.lower)
        left_tail_events = left_tail_events_mask.sum()
        left_cvar = self.label_returns[left_tail_events_mask].values.mean()
        right_tail_events_mask = (self.label_returns.values.squeeze() > self.upper)
        right_tail_events = right_tail_events_mask.sum()
        right_cvar = self.label_returns[right_tail_events_mask].values.mean()
        tail_events = left_tail_events + right_tail_events

        return pd.DataFrame({
            "first date": [dfpp.index[0]],
            "last date": [dfpp.index[-1]],
            "events": [len(dfpp)],
            "mse": [mse],
            "direction correct ratio": [direction_correct_ratio],
            "correlation": [corr],
            "r^2": [r2],
            "σ": [np.mean(self.predicted_std)],
            f"confidence (exp: {self.expected_confidence:.2f})": 1 - tail_events / len(dfpp),
            "left tail avg. distance %": [np.abs(self.lower.mean())],
            "right tail avg. distance %": [self.upper.mean()],
            "left tail events": left_tail_events / len(dfpp),
            "right tail events": right_tail_events / len(dfpp),
            "left cvar": left_cvar,
            "right cvar": right_cvar,
        }).T


class PriceSampledSummary(Summary):

    @staticmethod
    def with_reconstructor(
            label_returns: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y,
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame] = lambda y: y.ta.cumret(),
            sampler: Callable[[pd.Series], float] = lambda params, samples: np.random.normal(*params.values, samples),
            **kwargs):
        return lambda df, model, **kwargs2: PriceSampledSummary(
            df,
            model,
            label_returns,
            label_reconstruction,
            sampler,
            **{**kwargs, **kwargs2}
        )

    def __init__(
            self,
            df: Typing.PatchedDataFrame,
            model: Model,
            label_returns: Callable[[pd.DataFrame], pd.DataFrame],
            label_reconstruction: Callable[[pd.DataFrame], pd.DataFrame],
            sampler: Callable[[pd.Series], float] = lambda params, samples: np.random.normal(*params.values, samples),
            confidence: Union[float, Tuple[float, float]] = 0.8,
            forecast_period: int = 1,
            samples: int = 1000,
            bins='sqrt',
            figsize=(16, 16),
            **kwargs):
        super().__init__(
            df.sort_index(),
            model,
            self.plot_prediction,
            self.calc_scores,
            self.plot_tail_events,
            layout=[[0, 0],
                    [1, 2]],
            **kwargs
        )
        self.label_returns = call_callable_dynamic_args(label_returns, y=df[LABEL_COLUMN_NAME], df=df)
        self.label_reconstruction = call_callable_dynamic_args(label_reconstruction, y=self.label_returns, df=df)
        self.price_at_estimation = df[TARGET_COLUMN_NAME] if TARGET_COLUMN_NAME in df.columns else None

        self.sampler = sampler
        self.figsize = figsize
        self.forecast_period = forecast_period
        self.nr_samples = samples
        self.bins = bins

        # 0.8 => 0.1, 0.9
        self.left_confidence, self.right_confidence = \
            confidence if isinstance(confidence, Iterable) else ((1. - confidence) / 2, (1. - confidence) / 2 + confidence)
        self.expected_confidence = self.right_confidence - self.left_confidence

        self.cdf = self._estimate_ecdf()

    def _estimate_ecdf(self):
        params = self.df[PREDICTION_COLUMN_NAME].copy()
        params["samples"] = self.nr_samples

        return params.apply(lambda r: ECDF(self.sampler(r)), axis=1, result_type='expand')

    def plot_prediction(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from pandas_ml_quant.model.summary._utils import pandas_matplot_dates
        fig, ax = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        def confidence_band_price(cdf_last_price):
            cdf, last_price = cdf_last_price.values.tolist()
            min_max, alphas = cdf.heat_bar(self.bins)
            band = cdf.confidence_interval(self.left_confidence, self.right_confidence)
            return (min_max + 1) * last_price, alphas, (np.array(band) + 1) * last_price

        def confidence_band_return(cdf):
            cdf = cdf.item()
            min_max, alphas = cdf.heat_bar(self.bins)
            band = cdf.confidence_interval(self.left_confidence, self.right_confidence)
            return min_max, alphas, band

        def plot(ax, x, label, band, bar_edges, bar_colors, cmap=cm.BuPu_r, alpha=1):
            # plot label
            label.plot(ax=ax)

            # plot confidence
            ax.plot(x, band[:, 0], color='orange')
            ax.plot(x, band[:, 1], color='orange')
            ax.legend(['Label', 'Prediction'], loc='upper left')

            # plot distribution (with invisible bars)
            bars = ax.bar(x, bottom=bar_edges[:, 0], height=bar_edges[:, 1] - bar_edges[:, 0], width=1.0, color=None)
            lim = ax.get_xlim() + ax.get_ylim()

            # fill bars with heat like color
            for i, bar in enumerate(bars):
                bar.set_zorder(1)
                bar.set_facecolor("none")
                x, y = bar.get_xy()
                w, h = bar.get_width(), bar.get_height()
                c = np.flip(np.atleast_2d(bar_colors[i]).T)
                ax.imshow(c, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=cmap, alpha=alpha)

            # reset axis
            ax.axis(lim)
            # ax.colorbar(cm.ScalarMappable(cmap=cmap))
        try:
            # plot price chart
            price_data = pd.concat([self.cdf, self.label_reconstruction.shift(self.forecast_period)], axis=1)\
                .dropna()\
                .apply(confidence_band_price, axis=1, result_type='expand')\
                .join(self.label_reconstruction)

            x = pandas_matplot_dates(price_data)
            price_label = price_data.iloc[:, -1]
            price_band = price_data.iloc[:, -2]._.values
            bar_edges = price_data.iloc[:, 0]._.values
            bar_colors = price_data.iloc[:, 1]._.values

            plot(ax[0], x, price_label, price_band, bar_edges, bar_colors, cmap=cm.YlOrRd, alpha=0.4)

            # plot return chart
            return_data = self.cdf.to_frame()\
                .dropna() \
                .apply(confidence_band_return, axis=1, result_type='expand') \
                .join(self.label_returns)

            x = pandas_matplot_dates(return_data)
            return_label = return_data.iloc[:, -1]
            return_band = return_data.iloc[:, -2]._.values
            bar_edges = return_data.iloc[:, 0]._.values
            bar_colors = return_data.iloc[:, 1]._.values

            plot(ax[1], x, return_label, return_band, bar_edges, bar_colors, cmap=cm.YlOrRd, alpha=0.4)
        except Exception as e:
            traceback.print_exc()

        return fig

    def get_tail_events(self, clip=True):
        dfcdf = self.cdf.to_frame().dropna()
        idx = dfcdf.index.intersection(self.label_returns.index)
        dfl = self.label_returns.loc[idx]

        tail_events = dfcdf.join(dfl).apply(
            lambda x: x.iloc[0].get_tail_distance(x.iloc[1], self.left_confidence, self.right_confidence),
            axis=1,
            result_type='expand')

        tail_events.columns = ["left", "right"]
        return tail_events.clip(upper=0).replace(0, np.nan) if clip else tail_events

    def plot_tail_events(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        if "model" in kwargs: del kwargs["model"]

        try:
            ax = self.get_tail_events().hist(bins='sqrt', sharex=True, sharey=True, **kwargs)
            return plt.gcf()
        except Exception as e:
            traceback.print_exc()
            return None

    def calc_scores(self, *args, **kwargs):
        dfcdf = self.cdf.to_frame().dropna()
        idx = dfcdf.index.intersection(self.label_returns.index)
        dfl = self.label_returns.loc[idx]
        dfpp = self.cdf[idx].apply(lambda cdf: cdf.extreme())
        mean_std = self.cdf[idx].apply(lambda cdf: cdf.std()).mean()
        nr_events = len(dfcdf)

        direction_correct_ratio = \
            (((dfl.values.squeeze() > 0) & (dfpp.values.squeeze() > 0)) | ((dfl.values.squeeze() < 0) & (dfpp.values.squeeze() < 0))).sum() / len(dfl)

        corr, _ = pearsonr(dfl.values.flatten(), dfpp.values.flatten())
        r2 = r2_score(dfl, dfpp)

        tail_events = dfcdf.join(dfl).apply(
            lambda x: x.iloc[0].is_tail_event(x.iloc[1], self.left_confidence, self.right_confidence),
            axis=1,
            result_type='expand')

        cvars = dfcdf.apply(
            lambda cdf: cdf.iloc[0].cvar(self.left_confidence, self.right_confidence),
            axis=1,
            result_type='expand')

        # how many % is the confidence band away from the price at the day of prediction (the smaller the better)
        distance = dfcdf.apply(
            lambda cdf: cdf.iloc[0].confidence_interval(self.left_confidence, self.right_confidence),
            axis=1,
            result_type='expand').mean()

        # how wide is the confidence interval, the smaller the better
        band_width = dfcdf.apply(
            lambda cdf: cdf.iloc[0].confidence_band_width(self.left_confidence, self.right_confidence),
            axis=1,
            result_type='expand').mean()

        return pd.DataFrame({
            "first date": [dfcdf.index[0]],
            "last date": [dfcdf.index[-1]],
            "events": [nr_events],
            "direction correct ratio of extreme": [direction_correct_ratio],
            "correlation of extreme": [corr],
            "r^2 of extreme": [r2],
            "mean(σ)": [mean_std],
            f"confidence (exp: {self.expected_confidence:.2f} %)": [1 - tail_events.values.sum().item() / nr_events],
            "conf width": [band_width],
            "left tail avg. distance %": [np.abs(distance.iloc[0])],
            "right tail avg. distance %": [distance.iloc[1]],
            "left tail events %": [tail_events.iloc[:, 0].sum() / nr_events],
            "right tail events %": [tail_events.iloc[:, 1].sum() / nr_events],
            "left cvar %": [cvars.iloc[:, 0].mean()],
            "right cvar %": [cvars.iloc[:, 1].mean()],
        }).T

