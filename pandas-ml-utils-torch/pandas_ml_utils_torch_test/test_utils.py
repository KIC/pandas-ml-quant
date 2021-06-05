from unittest import TestCase

import pandas as pd
import numpy as np
from numpy import float32

from pandas_ml_utils_torch.utils import wrap_applyable


class TestUtils(TestCase):

    def test_wrap_applyable(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [[1, 2, 3],[1, 2, 3],[1, 2, 3]],
            "c": [[[1, 2, 3]],[[1, 2, 3]],[[1, 2, 3]]],
        })

        np.testing.assert_array_almost_equal(
            df[["a"]].apply(wrap_applyable(lambda x: x), axis=1, raw=False),
            [1, 2, 3]
        )

        np.testing.assert_array_almost_equal(
            df[["a"]].apply(wrap_applyable(lambda x: x), axis=1, raw=True),
            [[1], [2], [3]]
        )

        np.testing.assert_array_almost_equal(
            df[["a", "a"]].apply(wrap_applyable(lambda x: x.sum()), axis=1, raw=False),
            [2, 4, 6]
        )

        np.testing.assert_array_almost_equal(
            df[["a", "a"]].apply(wrap_applyable(lambda x: x.sum()), axis=1, raw=True),
            [2, 4, 6]
        )

        np.testing.assert_array_almost_equal(
            df["a"].apply(wrap_applyable(lambda x: x)),
            [1, 2, 3]
        )

        np.testing.assert_array_almost_equal(
            df["b"].apply(wrap_applyable(lambda x: sum(x.shape), return_numpy=False)).values,
            [3, 3, 3]
        )

        np.testing.assert_array_almost_equal(
            df[["c"]].apply(wrap_applyable(lambda x: sum(x.shape), return_numpy=False), axis=1).values,
            [4, 4, 4]
        )

        np.testing.assert_array_almost_equal(
            df[["c"]].apply(wrap_applyable(lambda x: sum(x.shape), return_numpy=False), axis=1, raw=True).values,
            [4, 4, 4]
        )
