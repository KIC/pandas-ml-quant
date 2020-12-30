

class LiveLossPlot(object):

    def __init__(self):
        # initialize a plot make sure backend is not inline but nbAgg (backend.replace('notebook', 'nbAgg'))
        pass

    def __call__(self, epoch, fold, fold_epoch, loss, val_loss):
        pass

