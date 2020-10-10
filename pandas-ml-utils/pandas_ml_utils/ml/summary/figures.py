from typing import Callable

from pandas_ml_common.utils.numpy_utils import one_hot, clean_one_hot_classification
from pandas_ml_utils.constants import *
from sklearn.metrics import roc_curve, auc, mean_squared_error
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


def plot_feature_importance(df,
                       model,
                       loss_function: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = mean_squared_error,
                       **kwargs):
    from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels
    from pandas_ml_utils.ml.data.extraction import extract, extract_feature_labels_weights
    from pandas_ml_common import Sampler

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
    # TODO calculate r2 score and such and return as data frame
    #	metrics.explained_variance_score(y_true, …)	Explained variance regression score function
    #	metrics.max_error(y_true, y_pred)	max_error metric calculates the maximum residual error.
    #	metrics.mean_absolute_error(y_true, y_pred, *)	Mean absolute error regression loss
    #	metrics.mean_squared_error(y_true, y_pred, *)	Mean squared error regression loss
    #	metrics.mean_squared_log_error(y_true, y_pred, *)	Mean squared logarithmic error regression loss
    #	metrics.median_absolute_error(y_true, y_pred, *)	Median absolute error regression loss
    #	metrics.r2_score(y_true, y_pred, *[, …])	R^2 (coefficient of determination) regression score function.
    #	metrics.mean_poisson_deviance(y_true, y_pred, *)	Mean Poisson deviance regression loss.
    #	metrics.mean_gamma_deviance(y_true, y_pred, *)	Mean Gamma deviance regression loss.
    #	metrics.mean_tweedie_deviance(y_true, y_pred, *)	Mean Tweedie deviance regression loss.

    pass


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

