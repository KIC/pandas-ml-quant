import logging
import numpy as np
import pandas as pd
from typing import List, Iterable, Union, Tuple
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from .correlation_analysis import plot_correlation_matrix, _sort_correlation, _plot_heatmap

_log = logging.getLogger(__name__)


def feature_selection(df: pd.DataFrame,
                      label: pd.Series,
                      nr_top_features: int = 5,
                      correlation_threshold: float = 0.5,
                      minimum_features: int = 1,
                      lags: Iterable[int] = range(100),
                      show_plots: bool = True,
                      figsize: Tuple[int, int] = (12, 10)):
    """
    The *feature_selection* functionality helps you to analyze your features, filter out highly
    correlated once and focus on the most important features. This function also applies an
    auto regression and embeds and ACF plot.

    :param df: the DataFrame of features which are considered for the label prediction
    :param label: columne of your dependent variable
    :param nr_top_features: number of most important features you want to select
    :param correlation_threshold: threshold at which correlated features drop out
    :param minimum_features: number of features you want to keep even if they are highly correlated
    :param lags: iterable of lags you want to analyze as an AR process
    :param show_plots: whether to show plots or not
    :param figsize: size of the polots
    :return: None
    """
    df = df.dropna()
    label = label.dropna()
    index = df.index.intersection(label.index)

    df = df.loc[index]
    label = label.loc[index]
    N = len(df)
    assert N > 1, f"data is empty {len(df)}, {len(label)}"

    correlation_mx = _sort_correlation(df.corr())

    if show_plots:
        plot_correlation_matrix(df)

    print(correlation_mx[:1])
    features = correlation_mx

    # find the features most correlated to the label (if provided)
    if label is not None:
        estimators, importances, indices, imporant_feature_names = \
            __feature_importance(df.values, label.values, features.columns)

        if show_plots:
            __plot_feature_importance(len(features.columns), estimators, importances, indices, imporant_feature_names, figsize)

        print(f"Feature ranking:\n{imporant_feature_names.tolist()}")
        top_features = imporant_feature_names[:min(nr_top_features, len(imporant_feature_names))]
        features = df[top_features].corr()

        print(f"\nTOP {nr_top_features} features")
        print(features[:1])

    # then eliminate features with high correlation to each other
    features = _sort_correlation(features)
    while len(features) > minimum_features and __max_correlation(features) > correlation_threshold:
        index = np.unravel_index(__argmax_correlation(features), features.values.shape)
        print(f"\nfilter feature {features.columns[index[1]]}"
              f" with correlation {features[features.columns[index[1]]].loc[features.index[index[1]]]}"
              f" > {correlation_threshold}")

        features = features.drop(features.index[index[1]], axis=0)
        features = features.drop(features.columns[index[1]], axis=1)

    print("Features after correlation filer:")
    print(correlation_mx[features.columns][:1])

    if show_plots:
        _plot_heatmap(features, figsize)

    # make AR analysis of remaining features
    coefficients = []
    for feature in features.columns:
        if show_plots:
            __plot_acf(df[feature], lags, (figsize[0], int(figsize[1] / 3)))

        dff = df[[feature]]
        for lag in lags:
            dff = dff.join(df[feature].shift(lag), rsuffix=f'_{lag}')

        dff = dff.corr().dropna()
        acorr = dff.iloc[0]
        coefficients.append(acorr.values)
        print(acorr.sort_values(ascending=False)[:10])

    # find clusters of auto correlation coefficients over all features
    cl = len(coefficients)
    best_lags = np.array(coefficients).sum(axis=0)
    best_lags_i = (-best_lags).argsort()
    best_lags = [(i, f'{-best_lags[i] / cl:.2f}') for i in best_lags_i if abs(best_lags[i]) > (1.96 / np.sqrt(N - i) / cl)]
    print(f"best lags are\n{best_lags[1:]}")

    return top_features.to_list(), [l[0] for l in best_lags]


def __feature_importance(x, y, names, n_estimators=250):
    continuous = 'float' in (str(y.dtype))
    print(f"label is continuous: {continuous}")

    forest = ExtraTreesRegressor(n_estimators=n_estimators, random_state=0) if continuous \
             else ExtraTreesClassifier(n_estimators=n_estimators, random_state=0)

    forest.fit(x, y)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # return ranking
    return forest.estimators_, importances, indices, names[indices]


def __plot_feature_importance(nr_of_features, estimators, importances, indices, names, figsize):
    std = np.std([tree.feature_importances_ for tree in estimators], axis=0)

    # Plot the feature importances of the forest
    try:
        from matplotlib import pyplot as plt
        plt.figure(figsize=figsize)
        plt.title("Feature importances")
        plt.bar(range(nr_of_features), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(nr_of_features), names, rotation='vertical')
        plt.xlim([-1, nr_of_features])
        plt.show()
    except:
        return None


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


def __max_correlation(df):
    return (df.abs().values - np.identity(len(df.columns))).max()


def __argmax_correlation(df):
    return (df.abs().values - np.identity(len(df.columns))).argmax()