import pandas_ml_utils.html as html
from pandas_ml_common import Typing
from pandas_ml_common.utils.serialization_utils import plot_to_html_img


class Summary(object):
    """
    Summary objects a used to visually present the results of a `df.fit` fitted model or a `df.backtest`
    All implementations of `Summary` need to:
     * pass a `pd.DataFrame` to `super().__init()__`
     * implement `_repr_html_()`
    """

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', *args, **kwargs):
        self._df = df
        self.model = model
        self.args = args
        self.kwargs = kwargs

    @property
    def df(self):
        return self._df

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        figures = [arg(self.df, model=self.model) for arg in self.args]
        embedded_plots = [plot_to_html_img(f) for f in figures if str(type(f)) == "<class 'matplotlib.figure.Figure'>"]
        tables = [f for f in figures if str(type(f)) != "<class 'matplotlib.figure.Figure'>"]

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self, plots=embedded_plots, tables=tables)

