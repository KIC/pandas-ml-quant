from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pandas_ml_common import naive_splitter
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.sampling.cross_validation import PartitionedOnRowMultiIndexCV
from pandas_ml_common.sampling.sampler import Sampler, XYWeight
from pandas_ml_common_test.config import TEST_MUTLI_INDEX_ROW_DF, TEST_DF


class TestSampler(TestCase):

    def test_simple_sample(self):
        def check_test(test_data):
            self.assertEqual(0, len(test_data))

        sampler = Sampler(XYWeight(TEST_DF, None,  TEST_DF.tail()), splitter=None, after_fold=check_test)
        batches = list(sampler.sample_for_training())

        self.assertEqual(1, len(batches))
        batch1 = batches[0]

        self.assertEqual(5, len(batch1.x))
        self.assertEqual(5, len(batch1.weight))
        self.assertIsNone(batch1.y)
        self.assertIsNone(batch1.y)

    def test_simple_sample_split(self):
        def check_test(test_data):
            self.assertEqual(1, len(test_data))
            self.assertEqual(2, len(test_data[0].x))
            self.assertIsNone(test_data[0].y)

        sampler = Sampler(XYWeight(TEST_DF, None, TEST_DF.tail()), splitter=lambda i, *args: (i[:3], i[3:]), epochs=2, after_fold=check_test)
        batches = list(sampler.sample_for_training())

        self.assertEqual(2, len(batches))
        self.assertEqual(3, len(batches[0].weight))
        self.assertIsNone(batches[0].y)

    def test_simple_batches(self):
        sampler = Sampler(
            XYWeight(TEST_DF, None, TEST_DF.tail()),
            splitter=lambda i: (i[:3], i[3:]),
            batch_size=2
        )

        batches = list(sampler.sample_for_training())

        self.assertEqual(2, len(batches))

        self.assertEqual(2, len(batches[0].x))
        self.assertEqual(2, len(batches[1].x))  # in case of one single instance add the 2nd last as well

    def test_multiple_test_data(self):
        def check_test(test_data):
            self.assertEqual(3, len(test_data))
            self.assertEqual('19930203', test_data[0].x.index[0].strftime("%Y%m%d"))
            self.assertEqual('19930202', test_data[0].x.index[1].strftime("%Y%m%d"))
            self.assertEqual('19930201', test_data[-1].x.index[0].strftime("%Y%m%d"))
            self.assertEqual('19930129', test_data[-1].x.index[1].strftime("%Y%m%d"))

        sampler = Sampler(
            XYWeight(TEST_DF),
            splitter=lambda i, *args: (i[:4], i[-3:]),
            cross_validation=lambda i: [([0, 1, 2], np.array([[3, 2, 1], [2, 1, 0]]))],
            after_fold=check_test
        )

        batches = list(sampler.sample_for_training())
        self.assertEqual(1, len(batches))
        self.assertEqual(3, len(batches[0].x))

    def test_multiframe_decorator(self):
        sampler = Sampler(
            XYWeight(MultiFrameDecorator((TEST_DF.tail(), TEST_DF)), TEST_DF.tail(10)),
            splitter=lambda i, *args: (i[:3], i[3:]),
            epochs=2
        )

        batches = list(sampler.sample_for_training())

        self.assertEqual(2, len(batches))
        self.assertEqual(3, len(batches[0].x))
        self.assertEqual(3, len(batches[1].y))

    def test_filter(self):
        sampler = Sampler(XYWeight(TEST_DF, TEST_DF),
                          splitter=lambda i, *args: (i[:-3], i[-3:]),
                          filter=lambda idx, y: y["Close"] > 300)
        batches = list(sampler.sample_for_training())

        self.assertEqual(1, len(batches))
        self.assertEqual(101, len(batches[0].x))
        self.assertEqual(101, len(batches[0].y))

    def test_cross_validation(self):
        def check_test(test_data):
            self.assertEqual(1, len(test_data))
            self.assertEqual(3, len(test_data[0]))

        sampler = Sampler(XYWeight(TEST_DF, TEST_DF, TEST_DF.tail(30)),
                          splitter=lambda i, *args: (i[:-3], i[-3:]),
                          cross_validation=KFold(3).split,
                          after_fold=check_test,
                          epochs=1)

        batches = list(sampler.sample_for_training())
        self.assertEqual(3, len(batches))
        batch1, batch2, batch3 = batches

        self.assertEqual(18, len(batch1.x))
        self.assertEqual(18, len(batch2.y))
        self.assertEqual(18, len(batch3.weight))

    def test_infinite_generator(self):
        sampler = Sampler(XYWeight(TEST_DF, None, TEST_DF.tail()), splitter=None, epochs=None)

        for i, s in enumerate(sampler.sample_for_training()):
            if i > 100:
                break

        self.assertGreater(i, 100)

    def test_simple_sample_split_multiindex_row(self):
        def check_test(test_data):
            self.assertIn("A", test_data[0].x.index)
            self.assertIn("B", test_data[0].x.index)

        sampler = Sampler(
            XYWeight(TEST_MUTLI_INDEX_ROW_DF),
            splitter=naive_splitter(0.5, partition_row_multi_index=True),
            after_fold=check_test,
            epochs=1
        )

        samples = list(sampler.sample_for_training())
        self.assertEqual(1, len(samples))
        self.assertEqual(4, len(samples[0].x))
        self.assertIn("A", samples[0].x.index)
        self.assertIn("B", samples[0].x.index)

    def test_cross_validation_multiindex_row(self):
        sampler = Sampler(
            XYWeight(TEST_MUTLI_INDEX_ROW_DF, TEST_MUTLI_INDEX_ROW_DF),
            splitter=None,
            cross_validation=PartitionedOnRowMultiIndexCV(KFold(2)),
            epochs=1
        )

        batches = list(sampler.sample_for_training())
        self.assertEqual(4, len(batches))

    def test_callback_early_stop(self):
        class Stopper(object):

            def __init__(self):
                self.i = 0

            def callback(self, epoch):
                self.i += 1
                if self.i > 2:
                    raise StopIteration

        sampler = Sampler(
            XYWeight(TEST_MUTLI_INDEX_ROW_DF, TEST_MUTLI_INDEX_ROW_DF),
            splitter=None,
            fold_epochs=100,
            after_fold_epoch=Stopper().callback
        )

        self.assertEqual(3, len(list(sampler.sample_for_training())))

    def test_fold_epoch_vs_epoch(self):
        df = pd.DataFrame({"a": np.arange(10)})
        fe = [d.x["a"].tolist() for d in Sampler(XYWeight(df[["a"]], df[["a"]]), fold_epochs=2).sample_for_training()]
        e = [d.x["a"].tolist() for d in Sampler(XYWeight(df[["a"]], df[["a"]]), epochs=2).sample_for_training()]
        self.assertListEqual(fe, e)