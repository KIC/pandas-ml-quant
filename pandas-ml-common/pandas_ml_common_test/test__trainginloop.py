from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_common.trainingloop import sampling
from pandas_ml_common import MlTypes, FeaturesLabels, naive_splitter, stratified_random_splitter
from pandas_ml_common_test.config import TEST_DF


class TestTrainingLoop(TestCase):

    def test_loop(self):
        """given a patched dataframe and a features and labels definition"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()
        fl = FeaturesLabels(features=["Open"], labels="Close")

        """when sampling from a sampler of the given dataframe and features"""
        _, sampler = sampling(df, fl, batch_size=len(df) // 2 + 1, epochs=2)
        batches = [batch for batch in sampler.sample_for_training()]

        """then we expect the result of a nice traing loop"""
        self.assertEqual(4, len(batches))
        self.assertEqual(1, len(batches[0].x))
        self.assertEqual(1, len(batches[0].y))
        self.assertEqual(len(df) * 2, sum([len(batches[i].x[0]) for i in range(0, 4)]))

    def test_loop_multiple_features(self):
        """given a patched dataframe and a features and labels definition for multiple features sets"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()
        fl = FeaturesLabels(features=[["Open"], ["High", "Low"]], labels=["Close"])

        """when sampling from a sampler of the given dataframe and features"""
        _, sampler = sampling(df, fl, batch_size=len(df) // 2 + 1, epochs=2)
        batches = [batch for batch in sampler.sample_for_training()]

        """then we expect the result of a nice traing loop"""
        self.assertEqual(4, len(batches))
        self.assertEqual(2, len(batches[0].x))
        self.assertEqual(1, len(batches[0].y))
        self.assertEqual(3413, len(batches[0].x[0]))
        self.assertEqual(len(df) * 2, sum([len(batches[i].x[0]) for i in range(0, 4)]))

    def test_loop_multirow_index(self):
        """given a patched dataframe and a features and labels definition for multiple features sets"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()
        df = pd.concat([df[:100], df[:100]], axis=0, keys=['A', 'B'])
        fl = FeaturesLabels(features=[["Open"], ["High", "Low"]], labels=["Close"])

        """when sampling from a sampler of the given dataframe and features"""
        _, sampler = sampling(df, fl, splitter=naive_splitter(partition_row_multi_index=True), batch_size=51, epochs=2)
        batches = [batch for batch in sampler.sample_for_training()]

        """then we expect the result of a nice traing loop"""
        self.assertEqual(6, len(batches))
        self.assertEqual(2, len(batches[0].x))
        self.assertEqual(1, len(batches[0].y))
        self.assertEqual(51, len(batches[0].x[0]))
        self.assertEqual({"A"}, set(batches[0].x[0].index.get_level_values(0)))
        self.assertEqual({"B"}, set(batches[-1].x[0].index.get_level_values(0)))

    def test_loop_multirow_index_split_batch(self):
        """given a patched dataframe and a features and labels definition for multiple features sets"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()
        df = pd.concat([df[:100], df[:100]], axis=0, keys=['A', 'B'])
        fl = FeaturesLabels(features=[["Open"], ["High", "Low"]], labels=["Close"])

        """when sampling from a sampler of the given dataframe and features"""
        _, sampler = sampling(df, fl, splitter=naive_splitter(partition_row_multi_index=True), batch_size=51, epochs=2, partition_row_multi_index_batch=True)
        batches = [batch for batch in sampler.sample_for_training()]

        """then we expect the result of a nice traing loop"""
        self.assertEqual(4 * 2, len(batches))
        self.assertEqual(2, len(batches[0].x))
        self.assertEqual(1, len(batches[0].y))
        self.assertEqual(51, len(batches[0].x[0]))
        for b in batches:
            self.assertEqual(1, len(set(b.x[0].index.get_level_values(0))))

    def test_loop_with_stratified_splitter(self):
        """given a patched dataframe and a features and labels definition for multiple features sets"""
        df = pd.DataFrame({"a": [True] * 100 + [False] * 900})
        fl = FeaturesLabels(features=["a"], labels=["a"])

        """when sampling from a sampler of the given dataframe and features"""
        _, sampler = sampling(df, fl, splitter=stratified_random_splitter(), batch_size=500, epochs=1)
        batches = [batch for batch in sampler.sample_for_training()]

        self.assertAlmostEqual(0.1, sum([b.y[0].sum().item() for b in batches]) / sum([len(b.y[0]) for b in batches]), places=2)
        self.assertAlmostEqual(0.1, df.loc[sampler.get_out_of_sample_features_index].sum().item() / len(sampler.get_out_of_sample_features_index), places=2)