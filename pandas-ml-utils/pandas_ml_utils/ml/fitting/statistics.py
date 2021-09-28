from collections import defaultdict

import pandas as pd

from .fitting_parameter import FittingParameter


class FitStatistics(object):

    def __init__(self, fitting_parameter: FittingParameter):
        self._history = defaultdict(dict)
        self.fitting_parameter = fitting_parameter

    def record_loss(self, epoch, fold, fold_epoch, train_loss, test_loss):
        # loss_history_key or fold
        self._history["train", ][(epoch, fold_epoch)] = train_loss
        self._history["test",fold][(epoch, fold_epoch)] = test_loss

    def plot_loss(self, figsize=(8, 6), **kwargs):
        """
        plot a diagram of the training and validation losses per fold
        :return: figure and axis
        """

        import matplotlib.pyplot as plt
        cm = 'tab20c'  # 'Pastel1'
        df = pd.DataFrame(self._history)
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))
        df['test'].plot(style='--', colormap=cm, ax=ax)
        df['train'].plot(colormap=cm, ax=ax)
        plt.legend(loc='upper right')
        return fig
