from typing import Callable, Tuple

from pandas_ml_common import MlTypes, call_callable_dynamic_args
from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from ..summary import Summary


class Fit(object):
    """
    After a model is fitted it gets embedded into this class along with some :class:`.Summary` statistics.
    In the case of `Fit` is displayed in a notebook the _repr_html of the Fit and Summary objects are called.
    """

    def __init__(self,
                 model: 'Fittable',
                 training_frames: FeaturesWithLabels,
                 test_frames: FeaturesWithLabels,
                 training_prediction: MlTypes.PatchedDataFrame,
                 test_prediction: MlTypes.PatchedDataFrame,
                 summary_provider: Callable[..., Summary],
                 **kwargs):
        self.model = model
        self.training_frames = training_frames
        self.test_frames = test_frames
        self.training_prediction = training_prediction
        self.test_prediction = test_prediction
        self.summary_provider = summary_provider
        self.kwargs = kwargs
        self._hide_loss_plot = False

    @property
    def prediction(self) -> Tuple[MlTypes.PatchedDataFrame, MlTypes.PatchedDataFrame]:
        return self.training_prediction, self.test_prediction

    @property
    def training_summary(self) -> Summary:
        return call_callable_dynamic_args(
            self.summary_provider,
            df=self.training_prediction,
            model=self.model,
            source=self.training_frames,
            **self.kwargs
        )

    @property
    def test_summary(self) -> Summary:
        return call_callable_dynamic_args(
            self.summary_provider,
            df=self.test_prediction,
            model=self.model,
            source=self.test_frames,
            **self.kwargs
        )

    def plot_loss(self, figsize=(8, 6), **kwargs):
        return self.model.fit_statistics.plot_loss(figsize, **kwargs)

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

    def with_summary(self, summary_provider: Callable[..., Summary] = Summary, **kwargs):
        return Fit(
            self.model,
            self.training_frames,
            self.test_frames,
            self.training_prediction,
            self.test_prediction,
            summary_provider,
            **self.kwargs,
            **kwargs
        )

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
            loss_plot=plot_to_html_img(self.model.fit_statistics.plot_loss, **self.kwargs) if not self._hide_loss_plot else None
        )