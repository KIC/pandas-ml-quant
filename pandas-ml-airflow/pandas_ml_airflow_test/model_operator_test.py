import pandas as pd
import numpy as np
from unittest import TestCase, mock

from pandas_ml_airflow.model_operator import MlModelOperator
from pandas_ml_utils import Model, FeaturesAndLabels


class ModelOperatorTest(TestCase):

    @mock.patch.object(Model, 'predict')
    def test_ml_model_operator(self, model):
        df = pd.DataFrame({"a": np.arange(10)})

        # TODO find a clean way to mock this ....
        model.__dict__['predict'] = lambda *args: df
        model.__dict__['features_and_labels'] = FeaturesAndLabels(
            features=["a"],
            labels=["a"],
        )

        op = MlModelOperator(
            task_id='test_operator',
            dataframe_provider=lambda ctx: df,
            model=model
        )

        res = op.execute({})
        res = pd.DataFrame.from_dict(res)
        self.assertIsInstance(res.columns, pd.MultiIndex)
        self.assertEqual(res.values[-1, -1], 9)