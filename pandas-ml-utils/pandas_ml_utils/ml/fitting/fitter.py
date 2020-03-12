from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, Tuple, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

from pandas_ml_common.utils import loc_if_not_none, call_if_not_none, merge_kwargs
from pandas_ml_utils.ml.data.extraction import extract_feature_labels_weights, extract
from pandas_ml_utils.ml.data.splitting import train_test_split
from pandas_ml_utils.ml.fitting.fit import Fit
from pandas_ml_utils.ml.model import Model
from pandas_ml_utils.ml.summary import Summary

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hyperopt import Trials


def fit(df: pd.DataFrame,
        model_provider: Callable[[int], Model],
        test_size: float = 0.4,
        youngest_size: float = None,
        cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        test_validate_split_seed = 42,
        hyper_parameter_space: Dict = None,
        **kwargs
        ) -> Fit:
    """

    :param df: the DataFrame you apply this function to
    :param model_provider: a callable which provides a new :class:`.Model` instance i.e. for each hyper parameter if
                           hyper parameter tuning is enforced. Usually all the Model subclasses implement __call__
                           thus they are a provider of itself
    :param test_size: the fraction [0, 1] of random samples which are used for a test set
    :param youngest_size: the fraction [0, 1] of the test samples which are not random but are the youngest
    :param cross_validation: tuple of number of epochs for each fold provider and a cross validation provider
    :param test_validate_split_seed: seed if train, test splitting needs to be reproduceable. A magic seed 'youngest' is
                                     available, which just uses the youngest data as test data
    :param hyper_parameter_space: space of hyper parameters passed as kwargs to your model provider
    :return: returns a :class:`pandas_ml_utils.model.fitting.fit.Fit` object
    """

    trails = None
    model = model_provider()
    kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
    features, labels, weights = extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)

    start_performance_count = perf_counter()
    _log.info("create model")

    # get indices and make training and test data sets
    train_idx, test_idx = train_test_split(features.index, test_size, youngest_size, test_validate_split_seed)
    train = (features.loc[train_idx], labels.loc[train_idx], loc_if_not_none(weights, train_idx))
    test = (features.loc[test_idx], labels.loc[test_idx], loc_if_not_none(weights, test_idx))

    # eventually perform a hyper parameter optimization first
    if hyper_parameter_space is not None:
        # next isolate hyperopt parameters and constants only used for hyper parameter tuning like early stopping
        constants = {}
        hyperopt_params = {}
        for k, v in list(hyper_parameter_space.items()):
            if k.startswith("__"):
                hyperopt_params[k[2:]] = hyper_parameter_space.pop(k)
            elif isinstance(v, (int, float, bool)):
                constants[k] = hyper_parameter_space.pop(k)

        # optimize hyper parameters
        model, trails = __hyper_opt(hyper_parameter_space,
                                    hyperopt_params,
                                    constants,
                                    model_provider,
                                    cross_validation,
                                    train,
                                    test)

    # finally train the model with eventually tuned hyper parameters
    __train_loop(model, cross_validation, train, test)
    _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

    # assemble result objects
    # FIXME add the logic back to the prediction data frame
    df_train = features_and_labels.prediction_to_frame(model.predict(train[0].ml.values), index=train[0].index, inclusive_labels=True)
    df_test = features_and_labels.prediction_to_frame(model.predict(test[0].ml.values), index=test[0].index, inclusive_labels=True) \
        if len(test[0]) > 0 else None

    # return the fit
    return Fit(model, model.summary_provider(df_train), model.summary_provider(df_test), trails)


def __train_loop(model, cross_validation, train, test):
    x_train, y_train, w_train = train[0].ml.values, train[1].ml.values, call_if_not_none(train[2], "values")
    x_test, y_test, w_test = test[0].ml.values, test[1].ml.values, call_if_not_none(test[2], "values")

    # apply cross validation
    if cross_validation is not None and isinstance(cross_validation, Tuple) and callable(cross_validation[1]):
        losses = []
        for fold_epoch in range(cross_validation[0]):
            # cross validation, make sure we re-shuffle every fold_epoch
            for f, (train_idx, test_idx) in enumerate(cross_validation[1](x_train, y_train)):
                _log.info(f'fit fold {f}')
                loss = model.fit(x_train[train_idx], y_train[train_idx],
                                 x_train[test_idx], y_train[test_idx],
                                 *((w_train[train_idx], w_train[test_idx]) if w_train is not None else (None, None)))

                losses.append(loss)

        return np.array(losses).mean()
    else:
        # fit without cross validation
        return model.fit(x_train, y_train, x_test, y_test, w_train, w_test)


@ignore_warnings(category=ConvergenceWarning)
def __hyper_opt(hyper_parameter_space,
                hyperopt_params,
                constants,
                model_provider,
                cross_validation,
                train,
                test):
    from hyperopt import fmin, tpe, Trials

    keys = list(hyper_parameter_space.keys())

    def f(args):
        sampled_parameters = {k: args[i] for i, k in enumerate(keys)}
        model = model_provider(**join_kwargs(sampled_parameters, constants))
        loss = __train_loop(model, cross_validation, train, test)
        if loss is None:
            raise ValueError("Can not hyper tune if model loss is None")

        return {'status': 'ok', 'loss': loss, 'parameter': sampled_parameters}

    trails = Trials()
    fmin(f, list(hyper_parameter_space.values()), algo=tpe.suggest, trials=trails, show_progressbar=False, **hyperopt_params)

    # find the best parameters amd make sure to NOT pass the constants as they are only used for hyperopt
    best_parameters = trails.best_trial['result']['parameter']
    best_model = model_provider(**best_parameters)

    print(f'best parameters: {repr(best_parameters)}')
    return best_model, trails


def predict(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    min_required_samples = model.features_and_labels.min_required_samples

    if tail is not None:
        if min_required_samples is not None:
            # just use the tail for feature engineering
            df = df[-(abs(tail) + (min_required_samples - 1)):]
        else:
            _log.warning("could not determine the minimum required data from the model")

    features_and_labels = FeatureTargetLabelExtractor(df, model.features_and_labels, **model.kwargs)
    x = features_and_labels.features_df
    y_hat = model.predict(x.feature_values)

    return features_and_labels.prediction_to_frame(y_hat, index=x.index, inclusive_labels=False)


def backtest(df: pd.DataFrame, model: Model, summary_provider: Callable[[pd.DataFrame], Summary] = Summary) -> Summary:
    features_and_labels = FeatureTargetLabelExtractor(df, model.features_and_labels, **model.kwargs)

    # make training and test data sets
    x = features_and_labels.features_df
    y_hat = model.predict(x.feature_values)

    df_backtest = features_and_labels.prediction_to_frame(y_hat, index=x.index, inclusive_labels=True, inclusive_source=True)
    return (summary_provider or model.summary_provider)(df_backtest)


