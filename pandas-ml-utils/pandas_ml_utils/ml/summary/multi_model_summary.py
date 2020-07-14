from pandas_ml_common import Typing
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.model.multi_model import MultiModel
from pandas_ml_utils.ml.summary import Summary


class MultiModelSummary(Summary):

    @staticmethod
    def filter_labels(df, model, nodel_nr):
        columns = [TARGET_COLUMN_NAME, PREDICTION_COLUMN_NAME, LABEL_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME, SAMPLE_WEIGHTS_COLUMN_NAME, FEATURE_COLUMN_NAME]
        columns = [col for col in columns if col in df]
        filtered_columns = []

        for col in columns:
            if col != FEATURE_COLUMN_NAME:
                cols = df[[col]].columns.to_list()
                column_slice = model._slices(nodel_nr, len(cols))
                filtered_columns = [*filtered_columns, *cols[column_slice]]
            else:
                filtered_columns = [*filtered_columns, *df[[FEATURE_COLUMN_NAME]].columns.to_list()]

        return filtered_columns

    def __init__(self, df: Typing.PatchedDataFrame, model: MultiModel, summary_provider=None, **kwargs):
        super().__init__(df, model, **kwargs)
        self.summaries = \
            [(summary_provider or sm.summary_provider)(df[MultiModelSummary.filter_labels(df, model, i)], sm, **kwargs)
             for i, sm in enumerate(model.sub_models)]

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(fit=self, summaries=self.summaries)
