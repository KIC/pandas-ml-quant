import os
from typing import Any

from pandas_ml_common import pd
from pandas_ml_common.serialization_utils import plot_to_html_img
from pandas_ml_utils.ml.model import Model
from pandas_ml_utils.ml.summary import Summary
import pandas_ml_utils.html as html


class Fit(object):
    """
    After a model is fitted it gets embedded into this class along with some :class:`.Summary` statistics.
    In the case of `Fit` is displayed in a notebook the _repr_html of the Fit and Summary objects are called.
    """

    def __init__(self,
                 model: Model,
                 training_summary: Summary,
                 test_summary: Summary,
                 trails: Any):
        self.model = model
        self.training_summary = training_summary
        self.test_summary = test_summary
        self._trails = trails

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

    def __str__(self):
        return f"train:\n" \
               f"{self.training_summary}" \
               f"\ntest:\n" \
               f"{self.test_summary}"

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.FIT_TEMPLATE, lookup=TemplateLookup(directories=['/']))
        return template.render(fit=self, loss_plot=plot_to_html_img(self.model.plot_loss))