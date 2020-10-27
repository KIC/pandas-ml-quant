from collections import defaultdict
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, mean_squared_error

from pandas_ml_common.utils import get_correlation_pairs, unique_level_rows, call_callable_dynamic_args
from pandas_ml_common.utils.numpy_utils import clean_one_hot_classification
from pandas_ml_utils.constants import *


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


def plot_feature_correlation(df, model=None, cmap='seismic', width=15, **kwargs):
    import matplotlib.pyplot as plt

    # extract features or use whole data frame
    if FEATURE_COLUMN_NAME in df:
        df = df[FEATURE_COLUMN_NAME]

    # cet correlations
    corr, sorted_corr = get_correlation_pairs(df)

    # generate plot
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(width, width / 1.33), gridspec_kw={'width_ratios': [3, 1]})
    ax.pcolor(corr, cmap=cmap)
    for i, j in np.ndindex(corr.shape):
        ax.text(j + 0.5, i + 0.5, f'{corr.iloc[i, j]:.2f}', ha="center", va="center", color="w")

    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(0.5, len(corr), 1))
    ax.set_yticklabels(corr.index)
    ax.set_xticks(np.arange(0.5, len(corr), 1))
    ax.set_xticklabels(corr.columns, rotation=60, ha="left", rotation_mode="anchor")

    ax2.pcolor(sorted_corr.values.reshape(-1, 1), cmap=cmap)
    for i, v in enumerate(sorted_corr.values):
        ax2.text(0.5, i + 0.5, f'{v:.2f}', ha="center", va="center", color="w")

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([])
    ax2.yaxis.tick_right()
    ax2.set_yticks(np.arange(0.5, len(sorted_corr), 1))
    ax2.set_yticklabels([f'{a} / {b}' for a, b in sorted_corr.index])

    return fig


def plot_partial_dependence(df, model, **kwargs):
    """
    def __plot_acf(series, lags, figsize):
    try:
        from matplotlib import pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(series.name, fontsize=14)

        plot_acf(series, lags=lags, ax=ax)
        plt.show()
    except:
        return None

    :param df:
    :param model:
    :param kwargs:
    :return:
    """
    raise NotImplemented


def plot_feature_importance(df,
                       model,
                       loss_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = mean_squared_error,
                       **kwargs):
    from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels
    from pandas_ml_utils.ml.data.extraction import extract, extract_feature_labels_weights

    frames: FeaturesWithLabels = extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)
    features, labels, targets, weights, gross_loss, latent = frames

    df_pred = df[PREDICTION_COLUMN_NAME]
    df_true = df[LABEL_COLUMN_NAME]

    loss_star = loss_function(df_pred.values, df_true.values, weights.values)
    # TODO implement the algorithm
    # for each feature  # FIXME how can we loop non lagged features? or should we just loop evey lag individually?
    #    shuffled_features = df[features].copy()
    #    do some shuffling
    #      features[feature] = np.roll(features[feature], len(features[feature]) // 2)
    #    sampler = Sampler(shuffled_features, labels, None, weights, None, None, splitter=None)
    #     df_pred = model.predict(sampler)
    #

    #  loss = df.model[feature].predict(features)  # uuu here it gets tricky ... we need to bypass the feature
    #                                              # extraction of the fitter
    # plot(loss - loss_star)

    # frames: FeaturesWithTargets = extract(model.features_and_labels, df, extract_features, **kwargs)
    #
    # if samples > 1:
    #     print(f"draw {samples} samples")
    #
    # # features, labels, targets, weights, gross_loss, latent,
    # sampler = Sampler(frames.features, None, frames.targets, None, None, frames.latent, splitter=None, epochs=samples)
    # predictions = model.predict(sampler, **kwargs)

    pass


def df_tail(df, model, **kwargs):
    # TODO if multiindex row make sure each level 0 element gets printed
    return df.tail()


def df_regression_scores(df, model, **kwargs):
    rm = metrics._regression

    def score_regression(df):
        y_true = df[LABEL_COLUMN_NAME]._.values
        y_pred = df[PREDICTION_COLUMN_NAME]._.values
        sample_weights = df[SAMPLE_WEIGHTS_COLUMN_NAME] if SAMPLE_WEIGHTS_COLUMN_NAME in df else None
        scores = defaultdict(lambda: [])

        for scorer in rm.__ALL__:
            try:
                score = call_callable_dynamic_args(rm.__dict__[scorer], y_true, y_pred, sample_weight=sample_weights)
                scores[scorer].append(score)
            except Exception as e:
                scores[scorer].append(str(e))

        return pd.DataFrame(scores)

    return score_regression(df) if not isinstance(df.index, pd.MultiIndex) else pd.concat(
        [score_regression(df.loc[group]).add_multi_index(group, inplace=True, axis=0) for group in unique_level_rows(df)],
        axis=0
    )

def df_classification_scores(df, model, **kwargs):
    # TODO calculate f1 score and such and return as data frame
    #  binary:
    #   average_precision_score(y_true, y_score, *)     Compute average precision(AP) from prediction scores
    #  multi class: Multiclass classification: classification task with more than two classes.
    #  Each sample can only be labelled as one class.
    #	balanced_accuracy_score(y_true, y_pred, *[, …])	Compute the balanced accuracy
    #	cohen_kappa_score(y1, y2, *[, labels, …])	Cohen’s kappa: a statistic that measures inter-annotator agreement.
    #	confusion_matrix(y_true, y_pred, *[, …])	Compute confusion matrix to evaluate the accuracy of a classification.
    #	hinge_loss(y_true, pred_decision, *[, …])	Average hinge loss (non-regularized)
    #	matthews_corrcoef(y_true, y_pred, *[, …])	Compute the Matthews correlation coefficient (MCC)
    #	roc_auc_score(y_true, y_score, *[, average, …])	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    #  Multilabel classification: classification task labelling each sample with x labels from n_classes possible
    #  classes, where x can be 0 to n_classes inclusive. This can be thought of as predicting properties of a sample
    #  that are not mutually exclusive.
    #	accuracy_score(y_true, y_pred, *[, …])	Accuracy classification score.
    #	classification_report(y_true, y_pred, *[, …])	Build a text report showing the main classification metrics.
    #	f1_score(y_true, y_pred, *[, labels, …])	Compute the F1 score, also known as balanced F-score or F-measure
    #	fbeta_score(y_true, y_pred, *, beta[, …])	Compute the F-beta score
    #	hamming_loss(y_true, y_pred, *[, sample_weight])	Compute the average Hamming loss.
    #	jaccard_score(y_true, y_pred, *[, labels, …])	Jaccard similarity coefficient score
    #	log_loss(y_true, y_pred, *[, eps, …])	Log loss, aka logistic loss or cross-entropy loss.
    #	multilabel_confusion_matrix(y_true, y_pred, *)	Compute a confusion matrix for each class or sample
    #	precision_recall_fscore_support(y_true, …)	Compute precision, recall, F-measure and support for each class
    #	precision_score(y_true, y_pred, *[, labels, …])	Compute the precision
    #	recall_score(y_true, y_pred, *[, labels, …])	Compute the recall
    #	roc_auc_score(y_true, y_score, *[, average, …])	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    #	zero_one_loss(y_true, y_pred, *[, …])	Zero-one classification loss.

    pass

