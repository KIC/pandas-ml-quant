from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

from pandas_ml_common import Typing, Sampler, XYWeight
from pandas_ml_common.utils.numpy_utils import clean_one_hot_classification
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from .figures import *
from .base_summary import Summary

_CONTEXT = "This summary should only be used by df.model.feature_selection()"


class FeatureSelectionSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, model: 'pandas_ml_utils.Model', is_test: bool, **kwargs):
        # assert the summary is used properly i.e. from the feature selection call
        assert hasattr(model, 'sk_model'), _CONTEXT
        assert isinstance(model.sk_model, RandomizedSearchCV), _CONTEXT
        assert isinstance(model.sk_model.best_estimator_, RFECV), _CONTEXT
        assert isinstance(model.sk_model.best_estimator_.estimator_, (ForestClassifier, ForestRegressor)), _CONTEXT

        # init super
        super().__init__(
            df,
            model,
            self._plot_feature_importance,
            plot_feature_correlation,
            lambda *args, **kwargs: self.kpis,
            layout=[
                [0, 0],
                [1, 2],
            ],
            **kwargs
        )

        # init self
        self.is_test = is_test
        self.nr_of_nodes = sum([e.tree_.node_count for e in model.sk_model.best_estimator_.estimator_.estimators_])

        def get_kpis(model):
            # originally passed features
            features = df[FEATURE_COLUMN_NAME].columns

            # get rankings
            rankings = model.sk_model.best_estimator_.ranking_  # best is least aka 1
            df_kpi = pd.DataFrame({"ranking": rankings, "importance": None, "importance_std": None}, index=features)
            selected_features = df_kpi[df_kpi["ranking"] <= 1].index

            # get importance of selected features
            importances = model.sk_model.best_estimator_.estimator_.feature_importances_
            importance_indices = np.argsort(importances)[::-1]

            # sorted kpi's by ranking
            importances = importances[importance_indices]
            importance_names = selected_features[importance_indices]
            importance_std = np.std(
                [tree.feature_importances_ for tree in model.sk_model.best_estimator_.estimator_.estimators_], axis=0)[
                importance_indices]

            # update the feature importance in then data frame
            df_kpi.loc[importance_names, "importance"] = importances
            df_kpi.loc[importance_names, "importance_std"] = importance_std

            return df_kpi

        # assign training data KPIs
        self.kpis = get_kpis(model)

        if is_test:
            from pandas_ml_utils.ml.fitting import FittingParameter

            # if this is the summary of the test set we should re-fit the model and finally compare the feature ranks
            features, labels = df[FEATURE_COLUMN_NAME], df[LABEL_COLUMN_NAME]
            weights = df[SAMPLE_WEIGHTS_COLUMN_NAME] if SAMPLE_WEIGHTS_COLUMN_NAME in df else None

            test_model = model()
            test_model.fit(XYWeight(features, labels, weights), FittingParameter(splitter=None, epochs=1))
            validation_kpis = get_kpis(test_model)

            self.kpis = self.kpis.join(validation_kpis, how='outer', rsuffix=" (Test)")

    def __str__(self):
        return str(self.kpis)

    def _plot_feature_importance(self, *args, figsize=(20, 10), **kwargs):
        fidf = self.kpis.fillna(0)

        if self.is_test:
            ax = fidf[["importance", "importance (Test)"]].plot.bar(
                yerr=fidf[["importance_std", "importance_std (Test)"]].values.T,
                figsize=figsize
            )
        else:
            ax = fidf[["importance", "importance_std"]].plot.bar(yerr='importance_std', figsize=figsize)

        ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=60, rotation_mode="anchor")
        return ax.get_figure()

        # PS the single version is:
        # fidf[["importances", "importance_std"]].plot.bar(yerr="importance_std")
