import numpy as np
import pandas as pd
from unittest import TestCase

from pandas_ml_utils.ml.data.splitting import RandomSequences, DummySplitter
from pandas_ml_utils.ml.data.splitting.sampeling import DataGenerator, Sampler
from pandas_ml_utils.ml.data.splitting.sampeling.extract_multi_model_label import ExtractMultiMultiModelSampler


class TestDataGenerator(TestCase):

    def test_generate_single_sample(self):
        data = pd.DataFrame({"a": np.arange(10)})
        test = (data, data, None)
        train = (data, data, None)
        gen = Sampler(train, test)

        samples = [s for s in gen.sample()]

        self.assertEqual(1, len(samples))

    def test_generate_multiple_sample(self):
        data = pd.DataFrame({"a": np.arange(10)})
        test = (data, data, None)
        train = (data, data, None)
        gen = Sampler(train, test, cross_validation=RandomSequences(0, 0.5, 12).cross_validation)

        samples = [s for s in gen.sample()]

        self.assertEqual(12, len(samples))

    def test_dummy_generator(self):
        features = pd.DataFrame({"a": np.arange(10)})
        targets = pd.DataFrame({"a": np.arange(10)})

        one = DataGenerator(DummySplitter(1), features, targets).complete_samples()
        two = DataGenerator(DummySplitter(2), features, targets).complete_samples()

        sampled1 = [s for s in one.sample()]
        sampled2 = [s for s in two.sample()]

        self.assertEqual(1, len(sampled1))
        self.assertEqual(2, len(sampled2))

    def test_wrapping_cross_validation(self):
        features = pd.DataFrame({"a": np.arange(10)})
        labels = pd.DataFrame({
            "a": np.arange(10),
            "b": np.arange(10) + 10,
            "c": np.arange(10) + 20,
            "d": np.arange(10) + 30,
        })

        s = DataGenerator(DummySplitter(2), features, labels, None, None, None).train_test_sampler()
        s1 = ExtractMultiMultiModelSampler(0, 2, s)
        s2 = ExtractMultiMultiModelSampler(1, 2, s)

        train, test = next(s.sample())
        y = train[1]
        self.assertEqual(4, y.shape[1])
        self.assertEqual(780, y.sum())

        train, test = next(s1.sample())
        y = train[1]
        self.assertEqual(2, y.shape[1])
        self.assertEqual(190, y.sum())

        train, test = next(s2.sample())
        y = train[1]
        self.assertEqual(2, y.shape[1])
        self.assertEqual(590, y.sum())

        for s in s1.sample():
            print("----")
            print(s[0][0])

        self.assertEqual(20, sum([len(s[0][0]) for s in s1.sample()]))
        self.assertEqual(20, sum([len(s[0][0]) for s in s2.sample()]))
