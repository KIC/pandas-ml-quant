import matplotlib.pyplot as plt
from IPython.display import clear_output

from pandas_ml_utils.ml.summary.regression_summary import RegressionSummary


def plot_losses(parent):

    class PlotLosses(parent):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            self.fig = plt.figure()
            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1

            clear_output(wait=True)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.show()

    return PlotLosses


def plot_true_pred_scatter(df, parent):

    class PlotTruePred(parent):
        def __init__(self):
            super().__init__()
            # FIXME
            #  1.) how to get train/test data
            #  2.) how to get a frame with label/prediction columns
            self.sum = RegressionSummary(df)

        def on_epoch_end(self, epoch, logs={}):
            clear_output(wait=True)
            self.sum.plot_true_pred_scatter()

    raise NotImplementedError() # TODO implement me ...
    return PlotTruePred

