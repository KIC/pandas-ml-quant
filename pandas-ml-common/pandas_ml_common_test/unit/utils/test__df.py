from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_common.utils import pd_hash


class TestDFUtils(TestCase):

    def test_hash(self):
        df = pd.DataFrame({
            "a": np.random.random(10),
            "b": np.random.random(10),
            "c": np.random.random(10),
        })

        df2 = df.copy()
        df2["b"] = np.random.random(10)

        self.assertNotEqual(pd_hash(df), pd_hash(df2))
        self.assertEqual(pd_hash(df, columns=["a", "c"]), pd_hash(df2, columns=["a", "c"]))
