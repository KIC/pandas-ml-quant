from typing import Callable

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils.time_utils import make_timeindex
from pandas_ml_quant.empirical import ECDF
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME
from pandas_ml_utils.ml.forecast import Forecast


class DistributionForecast(Forecast):

    @staticmethod
    def with_sampler(sampler: Callable[[Typing.PatchedSeries, int], np.ndarray], samples: int = 1000, *args, **kwargs):
        return lambda df: DistributionForecast(df, sampler, samples, *args, **kwargs)

    def __init__(self, df: Typing.PatchedDataFrame, sampler: Callable[[Typing.PatchedSeries, int], np.ndarray], samples: int = 1000, *args, **kwargs):
        super().__init__(self._sample(df, sampler, samples), *args, **kwargs)

    def hist(self, bins='sqrt') -> Typing.PatchedDataFrame:
        dfh = pd.concat([
            self.df[[('cdf', col)]]\
                .apply(lambda cdf: cdf.item().hist(bins=bins), axis=1, result_type='expand')
                .rename(columns={0: (col, "hist"), 1: (col, "edges")})
            for col in self.df['cdf'].columns
        ], axis=1)
        columns = pd.MultiIndex.from_tuples(dfh.columns)

        if TARGET_COLUMN_NAME in self.df.columns:
            dfh = self.df[TARGET_COLUMN_NAME]\
                      .join(dfh)\
                      .apply(lambda row: [v if name[1] == 'hist' else (v+1) * row[0] for name, v in row[1:].iteritems()],
                             axis=1,
                             result_type='expand')

        dfh.columns = columns
        return dfh

    def confidence_interval(self, lower: float = 0.1, upper: float = 0.9) -> Typing.PatchedDataFrame:
        ci_raw = pd.concat([
            self.df[[('cdf', col)]]\
                .apply(lambda cdf: cdf.item().confidence_interval(lower, upper), axis=1, result_type='expand')\
                .rename(columns={0: (col, "left"), 1: (col, "right")})
            for col in self.df['cdf'].columns
        ], axis=1)
        ci_raw.columns = pd.MultiIndex.from_tuples(ci_raw.columns)
        return (ci_raw + 1) * self.df[TARGET_COLUMN_NAME].values if TARGET_COLUMN_NAME in self.df.columns else ci_raw

    def forecast(
            self,
            forecast_period: int = 1,
            only_weekdays: bool = True,
            lower: float = 0.1,
            upper: float = 0.9,
            bins='sqrt'):
        prices = self.df[TARGET_COLUMN_NAME]
        price = prices.iloc[-1, 0]

        lower, upper = self.confidence_interval(lower, upper).iloc[-1].tolist()
        cone = pd.DataFrame({
            "lower": [(lower - price) * np.sqrt(i / forecast_period) + price for i in range(forecast_period + 1)],
            "upper": [(upper - price) * np.sqrt(i / forecast_period) + price for i in range(forecast_period + 1)]
        }, index=make_timeindex(self.df.index[-1], forecast_period, only_weekdays, include_start_date=True))

        hist, edges = self.hist(bins=bins).iloc[-1].tolist()
        histogram = pd.DataFrame({
            "hist": [hist] * (forecast_period + 1),
            "edges": [(edges - price) * np.sqrt(max(i, 0.1) / forecast_period) + price for i in range(forecast_period + 1)]
        }, index=cone.index)

        gaps = pd.DataFrame({}, index=pd.date_range(cone.index[0], cone.index[-1]))

        forecasts = pd.concat([cone, histogram, gaps], axis=1).ffill(axis=0)
        return pd.concat([prices, forecasts], axis=1)

    def plot_forecast(
            self,
            forecast_period: int = 1,
            only_weekdays: bool = True,
            lower: float = 0.1,
            upper: float = 0.9,
            figsize=(8, 6),
            title: str = "Forecast",
            cmap=None,
            alpha = 0.4,
            price_color='black',
            confidence_color='orange',
            fig_kwargs={}):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from matplotlib import cm

        fc = self.forecast(forecast_period, only_weekdays, lower, upper)

        # generate plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, **fig_kwargs)
        cmap = cm.YlOrRd if cmap is None else cmap
        fc.iloc[:, 0].plot(ax=ax, label='Price', color=price_color, legend=True)
        fc[["lower", "upper"]].plot(ax=ax, color=confidence_color, legend=False)

        # plot distribution (with invisible bars)
        fc = fc.loc[fc.iloc[:, 0].dropna().index[-1]:]
        bar_colors = fc["hist"]._.values
        bar_edges = fc["edges"]._.values
        bars = ax.bar(fc.index, bottom=bar_edges[:, 0], height=bar_edges[:, -1] - bar_edges[:, 0], width=1.0, color=None)

        # fill bars with heat like color
        lim = ax.get_xlim() + ax.get_ylim()
        for i, bar in enumerate(bars):
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            c = np.flip(np.atleast_2d(bar_colors[i]).T)
            ax.imshow(c, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=cmap, alpha=alpha)

        # reset axis
        ax.axis(lim)

        # xaxis and grid formating
        ax.set_title(title, y=1.0, pad=-17)
        ax.xaxis.set_major_locator(MaxNLocator(prune='both'))
        ax.xaxis.label.set_visible(False)
        ax.grid()

        return fig, ax

    def _sample(self, df, sampler, samples):
        pred = df[PREDICTION_COLUMN_NAME]

        def sample(s):
            return ECDF(sampler(s, samples))

        for col in pred.columns:
            cdf = pred[col].apply(sample)
            df = df.join(cdf.rename(('cdf', col)))

        return df