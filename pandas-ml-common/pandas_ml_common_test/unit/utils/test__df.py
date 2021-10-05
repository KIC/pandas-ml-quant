from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_common.utils import pd_hash, fix_multiindex_asymetry, pd_cumapply


class TestDFUtils(TestCase):

    def test_asymetry_fix(self):
        df = pd.DataFrame(np.random.random((3, 4)), pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 2)]))
        self.assertEqual((4, 4), fix_multiindex_asymetry(df.T.copy(), axis=1).shape)
        self.assertEqual((4, 4), fix_multiindex_asymetry(df, axis=0).shape)

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

    def test_cumapply(self):
        df = pd.DataFrame({"a": range(0, 10)})
        res = pd_cumapply(df, lambda cum, x: cum + x, start_value=12, axis=1)
        print(res)

        self.assertListEqual(
            [12, 13, 15, 18, 22, 27, 33, 40, 48, 57],
            res.iloc[:, 0].to_list()
        )
