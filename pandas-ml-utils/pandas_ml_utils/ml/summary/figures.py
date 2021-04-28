from collections import defaultdict
from collections import defaultdict
from itertools import chain
from typing import Callable
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, mean_squared_error

from pandas_ml_common.utils import get_correlation_pairs, unique_level_rows, call_callable_dynamic_args
from pandas_ml_common.utils.numpy_utils import clean_one_hot_classification
from pandas_ml_utils.constants import *

_log = logging.getLogger(__name__)


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


def plot_feature_correlation(df, model=None, cmap='bwr', width=15, **kwargs):
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


def plot_feature_importance(df, model, **kwargs):
    import matplotlib.pyplot as plt

    if FEATURE_COLUMN_NAME not in df:
        return f"DataFrame does not contain {FEATURE_COLUMN_NAME}!"

    mutation_steps = len(df) // 2
    dff = df[FEATURE_COLUMN_NAME]
    dfl = df[LABEL_COLUMN_NAME]
    dfw = df[SAMPLE_WEIGHTS_COLUMN_NAME] if SAMPLE_WEIGHTS_COLUMN_NAME in df else None
    loss_star = model.calculate_loss(None, dff, dfl, dfw)
    permuted_feature_loss = defaultdict(lambda: [])

    for col in dff.columns:
        x = dff.copy()
        x[col] = np.roll(x[col], mutation_steps)
        loss = model.calculate_loss(None, x, dfl, dfw)
        permuted_feature_loss[col].append(loss - loss_star)

    def sort_by_mean_error(item):
        return np.array(item[1]).mean()

    # more important features should cause a higher loss
    permuted_feature_loss = {k: v for k, v in sorted(permuted_feature_loss.items(), key=sort_by_mean_error, reverse=True)}

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    pd.DataFrame(permuted_feature_loss).plot.bar(ax=ax)
    ax.set_title('Featureimportance')
    return fig


def df_tail(df, model, **kwargs):
    return df.tail()


def df_regression_scores(df, model, **kwargs):
    rm = metrics._regression
    ALL = [
        "r2_score",
        "explained_variance_score",
        "mean_gamma_deviance",
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "mean_tweedie_deviance",
        "mean_poisson_deviance",
    ]

    def score_regression(df):
        y_true = df[LABEL_COLUMN_NAME]._.values
        y_pred = df[PREDICTION_COLUMN_NAME]._.values
        sample_weights = df[SAMPLE_WEIGHTS_COLUMN_NAME] if SAMPLE_WEIGHTS_COLUMN_NAME in df else None
        scores = defaultdict(lambda: [])

        for scorer in ALL:
            try:
                score = call_callable_dynamic_args(rm.__dict__[scorer], y_true, y_pred, sample_weight=sample_weights)
                scores[scorer].append(score)
            except Exception as e:
                _log.warning(f"{scorer} failed: {str(e)[:160]}")
                scores[scorer].append(np.nan)

        return pd.DataFrame(scores)

    fig_df = score_regression(df) if not isinstance(df.index, pd.MultiIndex) else pd.concat(
        [score_regression(df.loc[group]).add_multi_index(group, inplace=True, axis=0) for group in unique_level_rows(df)],
        axis=0
    )

    if kwargs.get("no_style", False):
        return fig_df.T

    try:
        # we try to apply styling for relative indicators if matplotlib is installed
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        cmap = plt.get_cmap('RdYlGn')

        def color_scaler(x, min, max, sign=1):
            out = (0, 1) if sign > 0 else (1, 0)
            try:
                col_val = np.interp(x, (min, max), out)
            except:
                col_val = out[0]

            return to_hex(cmap(col_val)) + "95"

        return fig_df.T.style\
            .apply(lambda x: [f"background-color: {color_scaler(x.item(), -1, 1)}"],
                   subset=pd.IndexSlice["r2_score", :]) \
            .apply(lambda x: [f"background-color: {color_scaler(x.item(), -1, 1)}"],
                   subset=pd.IndexSlice["explained_variance_score", :]) \
            .apply(lambda x: [f"background-color: {color_scaler(x.item(), 2, 0, -1)}"],
                   subset=pd.IndexSlice["mean_gamma_deviance", :]) \
            .set_precision(2)

    except:
        return fig_df.T


def df_classification_scores(df, model, **kwargs):
    class_scores = {  # [worst, best]
        "balanced_accuracy_score":  [0, 1],
        "cohen_kappa_score":    [0, 1],
        "matthews_corrcoef":    [0, 1],
        "f1_score": [0, 1],
        "jaccard_score":    [0, 1],
        "precision_score":  [0, 1],
        "recall_score": [0, 1],
        "accuracy_score":   [0, 1],
    }
    losses = {
        "roc_auc_score": [0.5, 1],
        "zero_one_loss": None,
        "hinge_loss": None,
        "log_loss": None,
        "hamming_loss": None,
    }

    def score_classification(df):
        y_true = df[LABEL_COLUMN_NAME]._.values
        y_pred = df[PREDICTION_COLUMN_NAME]._.values
        sample_weights = df[SAMPLE_WEIGHTS_COLUMN_NAME] if SAMPLE_WEIGHTS_COLUMN_NAME in df else None
        scores = defaultdict(lambda: [])

        y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 and y_true.shape[1] > 1 else y_true
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 and y_pred.shape[1] > 1 else y_pred > 0.5

        for scorer in class_scores.keys():
            try:
                score = call_callable_dynamic_args(metrics.__dict__[scorer], y_true_class, y_pred_class, sample_weight=sample_weights)
                if scorer == 'log_loss' and y_pred.ndim > 1:
                    score /= np.log(y_pred.shape[1])

                scores[scorer].append(score)
            except Exception as e:
                _log.warning(f"{scorer} failed: {str(e)[:160]}")
                scores[scorer].append(np.nan)

        for scorer in losses.keys():
            try:
                score = call_callable_dynamic_args(metrics.__dict__[scorer], y_true_class, y_pred, sample_weight=sample_weights)
                scores[scorer].append(score)
            except Exception as e:
                _log.warning(f"{scorer} failed: {str(e)[:160]}")
                scores[scorer].append(np.nan)

        return pd.DataFrame(scores)

    fig_df = score_classification(df) if not isinstance(df.index, pd.MultiIndex) else pd.concat(
        [score_classification(df.loc[group]).add_multi_index(group, inplace=True, axis=0) for group in unique_level_rows(df)],
        axis=0
    )

    if kwargs.get("no_style", False):
        return fig_df.T

    try:
        # we try to apply styling for relative indicators if matplotlib is installed
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        cmap = plt.get_cmap('RdYlGn')

        def color_scaler(x, min, max):
            col_val = np.interp(x, (min, max), (0, 1))
            return to_hex(cmap(col_val)) + "95"

        styled = fig_df.T.style.set_precision(2)
        for scorer, domain in chain(class_scores.items(), losses.items()):
            if domain is None:
                continue
            styled = styled.apply(lambda x, d=domain: [f"background-color: {color_scaler(x.item(), *d)}"],
                                  subset=pd.IndexSlice[scorer, :])

        return styled
    except Exception as e:
        return fig_df.T


