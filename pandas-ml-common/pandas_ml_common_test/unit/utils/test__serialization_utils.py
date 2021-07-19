from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_common.utils.serialization_utils import df_to_nested_dict


class TestSerializationUtils(TestCase):

    def test_nested_dict(self):
        df = pd.DataFrame({"a": np.random.rand(27)})
        df.index = pd.MultiIndex.from_tuples([(a, b, c) for a in range(3) for b in range(3) for c in range(3)])
        print(df_to_nested_dict(df))
