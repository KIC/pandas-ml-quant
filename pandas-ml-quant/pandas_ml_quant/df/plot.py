import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
from pandas.plotting import register_matplotlib_converters

from pandas_ml_common.plot import plot_bar, plot_stacked_bar, plot_candlestick, plot_line, plot_matrix
from pandas_ml_common.plot.animations import plot_animation

register_matplotlib_converters()


class TaPlot(object):

    def __init__(self, df: pd.DataFrame, figsize, rows=2, cols=1, main_height_ratio=4):
        fig = plt.figure('r-', figsize=figsize)
        grid = gridspec.GridSpec(rows, cols, height_ratios=[main_height_ratio, *[1 for _ in range(1, rows)]])
        axis = []

        for i, gs in enumerate(grid):
            ax = fig.add_subplot(gs, sharex=axis[0] if i > 0 else None)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

            if i < rows - 1:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', labelrotation=45)

            axis.append(ax)

        plt.xticks(rotation=45)

        self.df = df
        self.x = mdates.date2num(df.index)
        self.fig = fig
        self.axis = axis
        self.grid = grid

    def candlestick(self, open="Open", high="High", low="Low", close="Close", panel=0):
        self.axis[panel] = plot_candlestick(self.df, open, high, low, close, ax=self.axis[panel])
        return self._return()

    def stacked_bar(self, columns, padding=0.02, panel=1, **kwargs):
        # todo if x axis is multilevel then stack all bars at level > 2
        self.axis[panel] = plot_stacked_bar(self.df, columns, ax=self.axis[panel], padding=padding, **kwargs)
        return self._return()

    def bars(self):
        # FIXME add side by side bars
        # todo if x axis is multilevel then stack all bars at level > 2
        pass

    def bar(self, fields="Volume", panel=1, colors=None, color_map: str = 'afmhot', **kwargs):
        self.axis[panel] = plot_bar(self.df, fields, ax=self.axis[panel], colors=colors, color_map=color_map, **kwargs)
        return self._return()

    def line(self, fields="Close", panel=0, oscillator=False, **kwargs):
        self.axis[panel] = plot_line(self.df, fields, ax=self.axis[panel], **kwargs)
        return self._return()

    def plot_matrix(self, panel, fields, **kwargs):
        plot_matrix()

    def plot_matrix_animation(self, fields, fps=2, **kwargs):
        def make_frame(index):
            fig, ax = plt.subplots(figsize=(9, 9))
            plot_matrix(self.df.loc[[index]], fields, ax=ax, **kwargs)
            frame = mplfig_to_npimage(fig)
            plt.close(fig)
            return frame

        plt.close(self.fig)
        return plot_animation(self.df.index, make_frame, fps, **kwargs)

    def set_scale(self, panel, min, max):
        self.axis[panel].set_ylim(min, max)
        return self._return()

    def with_symetric_scale(self, *panels):
        for panel in panels:
            yl = np.abs(np.array(list(self.axis[panel].get_ylim()))).max() * 1.1
            self.axis[panel].set_ylim(-yl, +yl)

        return self._return()

    def __call__(self, *args, **kwargs):
        """
        TODO i am thinking of something like df.ta.plot()(candlesticks=True, sma=200)
             ... more thinking needed

        :param args:
        :param kwargs:
        :return:
        """
        if "lines" in kwargs:
            self.line(kwargs.pop('lines', None), **kwargs)
        else:
            self.line()

        if "bars" in kwargs:
            self.bar(kwargs.pop('bars', None), **kwargs)
        else:
            self.bar()

        return self.with_legend()

    def with_legend(self):
        for panel in self.axis:
            panel.legend(loc=2)

        return self

    def _return(self):
        self.grid.tight_layout(self.fig)




# %matplotlib
#inline
#
#from moviepy.editor import VideoClip
#from moviepy.video.VideoClip import DataVideoClip
#from moviepy.video.io.bindings import mplfig_to_npimage
#
#samples = gaf._.values[:, -1][-20:]
#
#
#def make_frame(sample):
#    fig, ax_patchwork = plt.subplots(figsize=(9, 9))
#
#    ax_patchwork.matshow(sample)
#    ax_patchwork.set_title("Gramian Angular Field")
#    ax_patchwork.set_yticklabels([])
#    ax_patchwork.set_xticklabels([])
#
#    frame = mplfig_to_npimage(fig)
#    plt.close(fig)
#
#    return frame
#
#
# # animation = VideoClip(make_frame, duration=5.1)
#animation = DataVideoClip(samples, make_frame, fps=4)
#
# # if(write_gif):
# # animation.write_gif("/tmp/foo.gif",fps=1)
#
#animation.ipython_display()
#