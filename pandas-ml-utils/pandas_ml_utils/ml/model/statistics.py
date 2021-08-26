from collections import defaultdict

import numpy as np
import pandas as pd
from typing import List

from pandas_ml_common import call_callable_dynamic_args, LazyInit, XYWeight


class ModelFitStatistics(object):

    def __init__(self):
        self._history = defaultdict(dict)

    def record_meta(self, epochs, batch_size, fold_epochs, cross_validation, features, labels: List[str], fitting_param):
        pass

    def record_loss(self, loss_history_key, epoch, fold, fold_epoch, train_loss, test_loss):
        # loss_history_key or fold
        self._history["train", ][(epoch, fold_epoch)] = train_loss
        self._history["test", loss_history_key or fold][(epoch, fold_epoch)] = test_loss

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
