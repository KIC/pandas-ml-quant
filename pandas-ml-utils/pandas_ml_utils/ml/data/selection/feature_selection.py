from typing import Union, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold

from pandas_ml_common import Typing as t
from pandas_ml_utils.ml.data.selection.feature_selection_report import FeatureSelectionReport, FeatureScoreReport


LINEAR = {
    "regressor": LinearRegression(),
    "classifier": LogisticRegression(multi_class='multinomial')
}

TREE = {
    "regressor": DecisionTreeRegressor(),
    "classifier": DecisionTreeClassifier()
}

MLP = {
    "regressor": MLPRegressor((30, 10), activation='tanh', early_stopping=True),
    "classifier": MLPClassifier((30, 10), activation='tanh', early_stopping=True)
}



def autoselect_features(df: t.PatchedDataFrame) -> FeatureSelectionReport:
    # TODO implement me
    return FeatureSelectionReport()


def score_feature(
        df,
        label,
        features,
        lags: Union[int, Iterable[int]] = [0],
        cv: _BaseKFold = KFold(6),
        regressor=DecisionTreeRegressor(),
        classifier=DecisionTreeClassifier()
    ) -> FeatureScoreReport:
    import matplotlib.pyplot as plt

    # get X and y data
    _X = df._[features]
    _y = df._[label]

    # convert categorical data
    if not (_X.dtype == np.float or _X.dtype == np.double):
        _X = _X.astype('category')
        _XEncoder = OneHotEncoder(drop='if_binary').fit(_X.values.codes.reshape(-1, 1))
    else:
        _XEncoder = None

    if not (_y.dtype == np.float or _y.dtype == np.double):
        _y = _y.astype('category')

    # prepare plots
    label_fig, label_ax = plt.subplots(cv.get_n_splits(), 1, figsize=(25, 10))
    label_fig.suptitle(_y.name)
    feature_lags_fig, feature_lags_ax = plt.subplots(1, 1 if _X.dtype.name == 'category' else 2, figsize=(25, 10))

    # estimate feature quality for each train test set and lag
    lags = list(range(lags)) if isinstance(lags, int) else lags

    # execute the following 2 times where we use pct_change() the 2nd time
    _XX = [_X] if _X.dtype.name == 'category' else [_X, _X.pct_change()]
    for xi, _X in enumerate(_XX):
        scores = []

        for li, lag in enumerate(lags):
            # lag data
            lag_score = []
            X = _X.shift(lag)
            y = _y

            # sync data
            idx = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[idx]
            y = y.loc[idx]

            # analyze feature quality
            for i, (train_index, test_index) in enumerate(cv.split(X)):
                train_y = y.iloc[train_index]
                test_y = y.iloc[test_index]

                # bring features in the expected shape and dummy code categorical variables
                if X.dtype.name == 'category':
                    encoded_x = _XEncoder.transform(X.values.codes.reshape(-1, 1)).toarray()
                else:
                    encoded_x = X.values.reshape(-1, 1)

                # add a constant trend such that we can regress vom trending to non trending vice versa
                encoded_x = np.concatenate([encoded_x, np.arange(len(encoded_x)).reshape(-1, 1)], axis=1)

                # perform linear or logistic regression
                if y.dtype.name == 'category':
                    if li == 0:
                        label_ax[i].scatter(train_y.index, train_y, c='b')
                        label_ax[i].scatter(test_y.index, test_y, c='r')

                    # One hot encode labels and perform regression
                    encoded_y = LabelEncoder().fit_transform(y.values.codes)
                    reg = classifier.fit(encoded_x[train_index], encoded_y[train_index])
                    y_hat = reg.predict(encoded_x[test_index])

                    # score model using f1
                    f1 = f1_score(encoded_y[test_index], y_hat, average=None).mean()
                    lag_score.append(f1)
                else:
                    if li == 0:
                        label_ax[i].plot(train_y, color='b')
                        label_ax[i].plot(test_y, color='r')

                    reg = regressor.fit(encoded_x[train_index], y.iloc[train_index].values)
                    y_hat = reg.predict(encoded_x[test_index])

                    # score model using r^2
                    r2 = r2_score(y.iloc[test_index], y_hat)
                    lag_score.append(r2)

            scores.append(lag_score)

        # add whisker plot for r^2 (or f1)
        feature_lags_fig.suptitle("f1" if _X.dtype.name == 'category' else 'r^2')
        ax = (feature_lags_ax[xi] if hasattr(feature_lags_ax, '__getitem__') else feature_lags_ax)
        ax.boxplot(scores, vert=False)
        ax.set_yticklabels([f'lag({l})' if xi == 0 else f'%(lag({l}))' for l in lags])
        ax.invert_yaxis()

    # return a summary of all plots and scores
    return FeatureScoreReport(label_fig, feature_lags_fig, scores)



# "^VIX", "Close"

#cv = TimeSeriesSplit(n_splits=5)
#cv = KFold(n_splits=5)
#lala(df, lambda df: df["^VIX", "Close"].shift(-1), [("^VIX", "Close")], cv)


