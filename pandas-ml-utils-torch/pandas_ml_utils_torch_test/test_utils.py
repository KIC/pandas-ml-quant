from unittest import TestCase

import numpy as np
import numpy.testing
import pandas as pd

from pandas_ml_utils_torch.utils import wrap_applyable


class TestUtils(TestCase):

    def test_wrap_applyable(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [[1, 2, 3],[1, 2, 3],[1, 2, 3]],
            "c": [[[1, 2, 3]],[[1, 2, 3]],[[1, 2, 3]]],
        })

        np.testing.assert_array_almost_equal(
            df[["a"]].apply(wrap_applyable(lambda x: x*1), axis=1, raw=False),
            [1, 2, 3]
        )

        np.testing.assert_array_almost_equal(
            df[["a"]].apply(wrap_applyable(lambda x: x*1), axis=1, raw=True),
            [1, 2, 3]
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
            df["a"].apply(wrap_applyable(lambda x: x*1)),
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

    def test_wrap_distribution(self):
        from torch.distributions import MixtureSameFamily, Categorical, Normal

        df = pd.DataFrame({
            "b": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            "c": [1, 2, 3],
        })

        def variant_a(tens):
            return MixtureSameFamily(
                Categorical(probs=tens[:1]),
                Normal(loc=tens[1:2], scale=tens[2:3])
            )

        def variant_b(tens, b):
            return MixtureSameFamily(
                Categorical(probs=tens[:1]),
                Normal(loc=tens[1:2], scale=tens[2:3])
            ).cdf(b)

        # variant a
        dists_a = df[["c"]]\
            .join(df["b"].apply(wrap_applyable(variant_a, return_numpy=False)))\
            .apply(wrap_applyable(lambda val, dist: dist.cdf(val), nr_args=2), axis=1)

        # variant b
        dists_b = df.apply(wrap_applyable(variant_b, nr_args=2), axis=1)

        numpy.testing.assert_array_almost_equal(dists_a, dists_b)