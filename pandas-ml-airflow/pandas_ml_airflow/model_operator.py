from typing import Callable, Dict

from airflow.operators.python_operator import PythonOperator

from pandas_ml_common import Typing
from pandas_ml_utils import Model


class MlModelOperator(PythonOperator):

    def __init__(self,
                 dataframe_provider: Callable[[Dict], Typing.PatchedDataFrame],
                 model: Model,
                 post_processor: Callable[[Dict, Typing.PatchedDataFrame, Typing.PatchedDataFrame], Typing.Pandas] = None,
                 predictions: int = 1,
                 **kwargs):
        super().__init__(**kwargs, provide_context=True, python_callable=self.predict)
        self.dataframe_provider = dataframe_provider
        self.model = model
        self.post_processor = post_processor
        self.predictions = predictions

    def predict(self, **context):
        df = self.dataframe_provider(context)
        self.log.info(f"fetched {len(df)} rows, columns: {df.columns}")

        pdf = df.model.predict(self.model, tail=self.predictions)

        if self.post_processor is not None:
            pdf = self.post_processor(context, df, pdf)

        return pdf.to_dict() if isinstance(pdf, (Typing.DataFrame, Typing.Series)) else pdf
