from pandas_ml_common.utils.numpy_utils import one_hot, clean_one_hot_classification
from pandas_ml_utils.constants import *
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_true_pred_scatter(df, figsize=(6, 6), alpha=0.1, **kwargs):
    import matplotlib.pyplot as plt

    x = df[PREDICTION_COLUMN_NAME].values
    y = df[LABEL_COLUMN_NAME].values
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())

    fig, axis = plt.subplots(figsize=figsize)
    plt.axes(aspect='equal')
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.scatter(x, y, alpha=alpha)
    plt.xlim([mn, mx])
    plt.ylim([mn, mx])
    plt.plot([mn, mx], [mn, mx])

    return fig


def plot_receiver_operating_characteristic(df, figsize=(6, 6), **kwargs):
    import matplotlib.pyplot as plt

    # get true and prediction data. It needs to be a one hot encoded 2D array [samples, class] where nr_classes >= 2
    tv, pv = clean_one_hot_classification(df[LABEL_COLUMN_NAME]._.values, df[PREDICTION_COLUMN_NAME]._.values)

    # calculate legends
    legend = [(col[1] if isinstance(col, tuple) else col) for col in df[LABEL_COLUMN_NAME].columns.tolist()]
    if tv.shape[1] > len(legend):
        legend = list(range(tv.shape[1]))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(tv.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(tv[:, i], pv[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot ROC curves
    fig, axis = plt.subplots(figsize=figsize)

    for i in fpr.keys():
        plt.plot(fpr[i], tpr[i], label=f"{legend[i]} auc:{roc_auc[i] * 100:.2f}")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return fig


def plot_confusion_matrix(df, figsize=(6, 6), **kwargs):
    from mlxtend.plotting import plot_confusion_matrix
    from mlxtend.evaluate import confusion_matrix

    # get true and prediction data. It needs to be a one hot encoded 2D array [samples, class] where nr_classes >= 2
    tv, pv = clean_one_hot_classification(df[LABEL_COLUMN_NAME]._.values, df[PREDICTION_COLUMN_NAME]._.values)

    # confusion matrix needs integer encoding
    tv = np.apply_along_axis(np.argmax, 1, tv)
    pv = np.apply_along_axis(np.argmax, 1, pv)

    # plot the confusion matrix
    cm = confusion_matrix(tv, pv, binary=tv.max() < 2)
    fig, ax = plot_confusion_matrix(cm, figsize=figsize)

    return fig


def feature_importance(df, model, **kwargs):
    # loss_star = model.predict() get loss
    # for each feature
    #  features = df[features].copy()
    #  features[feature] = np.roll(features[feature], len(features[feature]) // 2)
    #  loss = df.model[feature].predict(features)  # uuu here it gets tricky ... we need to bypass the feature
    #                                              # extraction of the fitter
    # plot(loss - loss_star)
    pass