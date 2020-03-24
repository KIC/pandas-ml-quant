import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas.core.base import PandasObject


def matplot_dates(df: PandasObject):
    return mdates.date2num(df.index)


def new_fig_ts_axis(figsize=(12, 8)):
    fig = plt.figure('r-', figsize=figsize)
    grid = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45)

    return fig, ax
