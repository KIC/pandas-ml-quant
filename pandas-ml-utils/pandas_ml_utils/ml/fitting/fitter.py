from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, Dict, TYPE_CHECKING

import pandas as pd

from pandas_ml_common.utils import merge_kwargs, to_pandas
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.data.extraction import extract_feature_labels_weights, extract, extract_features
from pandas_ml_utils.ml.data.reconstruction import assemble_prediction_frame
from pandas_ml_utils.ml.data.splitting import DummySplitter, Splitter
from pandas_ml_utils.ml.data.splitting.random_splits import RandomSplits
from pandas_ml_utils.ml.data.splitting.sampeling import DataGenerator
from pandas_ml_utils.ml.fitting.fit import Fit
from pandas_ml_utils.ml.model import Model
from pandas_ml_utils.ml.summary import Summary

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def fit(df: pd.DataFrame,
        model_provider: Callable[[int], Model],
        training_data_splitter: Splitter = RandomSplits(),
        hyper_parameter_space: Dict = None,
        **kwargs
        ) -> Fit:
    """

    :param df: the DataFrame you apply this function to
    :param model_provider: a callable which provides a new :class:`.Model` instance i.e. for each hyper parameter if
           hyper parameter tuning is enforced. Usually all the Model subclasses implement __call__ thus they are a
           provider of itself
    :param training_data_splitter: a :class:`pandas_ml_utils.ml.data.splitting.Splitter` object
           which provides traning and test data splits (eventually multiple folds)
    :param hyper_parameter_space: space of hyper parameters passed as kwargs to your model provider
    :return: returns a :class:`pandas_ml_utils.model.fitting.fit.Fit` object
    """

    trails = None
    model = model_provider()
    kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
    (features, min_required_samples), labels, targets, weights, gross_loss = \
        extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)

    start_performance_count = perf_counter()
    _log.info("create model")

    # get indices and make training and test data sets
    #train_idx, test_idx = training_data_splitter.train_test_split(features.index)
    #train = (features.loc[train_idx], labels.loc[train_idx], loc_if_not_none(weights, train_idx))
    #test = (features.loc[test_idx], labels.loc[test_idx], loc_if_not_none(weights, test_idx))

    # FIXME eventually perform a hyper parameter optimization first
    #if hyper_parameter_space is not None:
    #    # next isolate hyperopt parameters and constants only used for hyper parameter tuning like early stopping
    #    constants = {}
    #    hyperopt_params = {}
    #    for k, v in list(hyper_parameter_space.items()):
    #        if k.startswith("__"):
    #            hyperopt_params[k[2:]] = hyper_parameter_space.pop(k)
    #        elif isinstance(v, (int, float, bool)):
    #            constants[k] = hyper_parameter_space.pop(k)
    #
    #    # optimize hyper parameters
    #    model, trails = __hyper_opt(hyper_parameter_space,
    #                                hyperopt_params,
    #                                constants,
    #                                model_provider,
    #                                None, # FIXME Ecross_validation,
    #                                train,
    #                                test)

    # finally train the model with eventually tuned hyper parameters
    sampler = DataGenerator(training_data_splitter, features, labels, targets, weights, gross_loss).train_test_sampler()
    model.fit(sampler, **kwargs)
    _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

    # assemble result objects
    train_sampler, train_idx = sampler.training()
    test_sampler, test_idx = sampler.validation()
    prediction = (to_pandas(model.predict(train_sampler, **kwargs), train_idx, labels.columns),
                  to_pandas(model.predict(test_sampler, **kwargs), test_idx, labels.columns))

    # get training and test data tuples of the provided frames
    features, labels, targets, weights, gross_loss = sampler[0], sampler[1], sampler[2], sampler[3], sampler[4]
    df_train, df_test = [
        _assemble_result_frame(targets[i], prediction[i], labels[i], gross_loss[i], weights[i], features[i])
        for i in range(2)]

    # update model properties and return the fit
    model._validation_indices = test_idx
    model.features_and_labels.set_min_required_samples(min_required_samples)
    model.features_and_labels.set_label_columns(labels[0].columns.tolist())
    return Fit(model, model.summary_provider(df_train, **kwargs), model.summary_provider(df_test, **kwargs), trails, **kwargs)


def predict(df: pd.DataFrame, model: Model, tail: int = None, samples: int = 1, **kwargs) -> pd.DataFrame:
    min_required_samples = model.features_and_labels.min_required_samples

    if tail is not None:
        if min_required_samples is not None:
            # just use the tail for feature engineering
            df = df[-(abs(tail) + (min_required_samples - 1)):]
        else:
            _log.warning("could not determine the minimum required data from the model")

    kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
    columns, features, targets = extract(model.features_and_labels, df, extract_features, **kwargs)

    if samples > 1:
        print(f"draw {samples} samples")

    sampler = DataGenerator(DummySplitter(samples), features, None, targets, None).complete_samples()
    predictions = model.predict(sampler, **kwargs)

    y_hat = to_pandas(predictions, index=features.index, columns=columns)
    return _assemble_result_frame(targets, y_hat, None, None, None, features)


def backtest(df: pd.DataFrame, model: Model, summary_provider: Callable[[pd.DataFrame], Summary] = Summary, **kwargs) -> Summary:
    kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
    (features, _), labels, targets, weights, gross_loss =\
        extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)

    sampler = DataGenerator(DummySplitter(1), features, labels, targets, None).complete_samples()
    predictions = model.predict(sampler, **kwargs)

    y_hat = to_pandas(predictions, index=features.index, columns=labels.columns)
    df_backtest = _assemble_result_frame(targets, y_hat, labels, gross_loss, weights, features)
    return (summary_provider or model.summary_provider)(df_backtest, **kwargs)


def _assemble_result_frame(targets, prediction, labels, gross_loss, weights, features):
    return assemble_prediction_frame(
        {TARGET_COLUMN_NAME: targets,
         PREDICTION_COLUMN_NAME: prediction,
         LABEL_COLUMN_NAME: labels,
         GROSS_LOSS_COLUMN_NAME: gross_loss,
         SAMPLE_WEIGHTS_COLUMN_NAME: weights,
         FEATURE_COLUMN_NAME: features})



"""
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
        model = None # FIXME model_provider(**join_kwargs(sampled_parameters, constants))
        loss = None # FIXME __train_loop(model, cross_validation, train, test)
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

"""