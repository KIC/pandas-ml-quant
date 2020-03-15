import os
import pandas as pd
import pandas_ml_utils.html as html


class Summary(object):
    """
    Summary objects a used to visually present the results of a `df.fit` fitted model or a `df.backtest`
    All implementations of `Summary` need to:
     * pass a `pd.DataFrame` to `super().__init()__`
     * implement `_repr_html_()`
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        self._df = df
        self.kwargs = kwargs

    @property
    def df(self):
        return self._df

    def _html_template_file(self):
        return f"{os.path.abspath(__file__)}.html"

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self)

