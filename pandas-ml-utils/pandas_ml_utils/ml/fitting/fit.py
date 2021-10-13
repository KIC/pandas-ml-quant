from typing import Any, Callable

import pandas as pd


from pandas_ml_common import MlTypes
from pandas_ml_common.utils import merge_kwargs
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from ..summary import Summary


class Fit(object):
    """
    After a model is fitted it gets embedded into this class along with some :class:`.Summary` statistics.
    In the case of `Fit` is displayed in a notebook the _repr_html of the Fit and Summary objects are called.
    """

    def __init__(self,
                 model: 'Fittable',
                 source_df: MlTypes.PatchedDataFrame,
                 training_prediction_df: MlTypes.PatchedDataFrame,
                 test_prediction_df: MlTypes.PatchedDataFrame,
                 summary_provider: Callable[..., Summary],
                 trails: Any = None,
                 **kwargs):
        self.model = model
        self.source_df = source_df
        self.training_prediction_df = training_prediction_df
        self.test_prediction_df = test_prediction_df

        self.training_summary = Summary.provide(summary_provider, training_prediction_df, model, source_df, **kwargs)
        self.test_summary = Summary.provide(summary_provider, test_prediction_df, model, source_df, **kwargs)
        self._trails = trails
        self._kwargs = kwargs
        self._hide_loss_plot = False

    def plot_loss(self, figsize=(8, 6), **kwargs):
        return self.model.fit_statistics.plot_loss(figsize, **kwargs)

    def values(self):
        """
        :return: returns the fitted model, a :class:`.Summary` on the training data, a :class:`.Summary` on the test data
        """
        return self.model, self.training_summary, self.test_summary

    def trails(self):
        """
        In case of hyper parameter optimization a trails object as used by `Hyperopt <https://github.com/hyperopt/hyperopt/wiki/FMin>`_
        is available.

        :return: Trails object
        """
        if self._trails is not None:
            return pd.DataFrame(self._trails.results)\
                     .drop("parameter", axis=1)\
                     .join(pd.DataFrame([r['parameter'] for r in self._trails.results]))
        else:
            return None

    def save_model(self, filename: str):
        """
        Save the fitted model.

        :param filename: filename
        :return: None
        """
        self.model.save(filename)

    def with_hidden_loss_plot(self):
        self._hide_loss_plot = True
        return self

    def with_summary(self, summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = Summary, **kwargs):
        return Fit(self.model,
                   self.source_df,
                   self.training_prediction_df,
                   self.test_prediction_df,
                   summary_provider,
                   self._trails,
                   **merge_kwargs(self._kwargs, kwargs))

    def __str__(self):
        summaries = f"train:\n{self.training_summary}\ntest:\n{self.test_summary}"
        backend = None
        try:
            import matplotlib
            backend = matplotlib.get_backend()
            matplotlib.use('module://drawilleplot')
            fig = self.model.fit_statistics.plot_loss()
            return f"{fig}\n{summaries}"
        except:
            return summaries
        finally:
            if backend is not None:
                import matplotlib
                matplotlib.use(backend)

    def _repr_html_(self):
        import pandas_ml_utils.html as html
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.FIT_TEMPLATE, lookup=TemplateLookup(directories=['/']))
        return template.render(
            fit=self,
            loss_plot=plot_to_html_img(self.model.fit_statistics.plot_loss, **self._kwargs) if not self._hide_loss_plot else None
        )