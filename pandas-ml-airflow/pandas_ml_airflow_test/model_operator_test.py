import pandas as pd
import numpy as np
from unittest import TestCase

from pandas_ml_airflow.model_operator import MlModelOperator
from pandas_ml_utils import DummyModel, FeaturesAndLabels


class ModelOperatorTest(TestCase):

    def test_ml_model_operator(self):
        df = pd.DataFrame({"a": np.arange(10)})
        model = DummyModel(
            FeaturesAndLabels(
                features=["a"],
                labels=["a"],
            )
        )

        fit = df.model.fit(model)
        model = fit.model

        op = MlModelOperator(
            task_id='test_operator',
            dataframe_provider=lambda ctx: df,
            model=model
        )

        res = op.execute({})
        res = pd.DataFrame.from_dict(res)
        self.assertIsInstance(res.columns, pd.MultiIndex)
        self.assertEqual(res.values[-1, -1], 9)