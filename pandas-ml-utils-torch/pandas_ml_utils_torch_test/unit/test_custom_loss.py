from unittest import TestCase

import torch as t
import torch.nn as nn
from torch.optim import Adam

from pandas_ml_common.utils.column_lagging_utils import lag_columns
from pandas_ml_common.utils.numpy_utils import one_hot
from pandas_ml_common import np
from pandas_ml_common_test.config import TEST_DF
from pandas_ml_utils import FeaturesLabels, FittingParameter, FittableModel, AutoEncoderModel
from pandas_ml_utils_torch import PytorchAutoEncoderFactory, PytorchNN, PytorchModelProvider
from pandas_ml_utils_torch import lossfunction


class TestCustomLoss(TestCase):

    def test__differentiable_argmax(self):
        """given"""
        args = 10
        argmax = lossfunction.DifferentiableArgmax(args)

        """when"""

        res = np.array([argmax(t.tensor(one_hot(i, args))).numpy() for i in range(args)])

        """then"""
        print(res)
        np.testing.assert_array_almost_equal(res, np.arange(0, args))

    def test__parabolic_crossentropy(self):
        """when"""
        categories = 11
        loss_function = lossfunction.ParabolicPenaltyLoss(categories, 1)

        """then"""
        for truth in range(categories):
            losses = []
            for prediction in range(categories):
                loss = loss_function(t.tensor(one_hot(prediction, categories)),
                                     t.tensor(one_hot(truth, categories)))
                losses.append(loss)

            # all predictions left of truth need to increase
            for i in range(1, truth):
                self.assertGreater(losses[i - 1], losses[i])

            # right of truth need to decrease
            for i in range(truth, categories - 1):
                self.assertLess(losses[i], losses[i + 1])

            if truth > 0 and truth < categories - 1:
                if truth > categories / 2:
                    # right tail:
                    self.assertGreater(losses[truth - 1], losses[truth + 1])
                else:
                    # left tail
                    self.assertGreater(losses[truth + 1], losses[truth - 1])

    def test__tailed_categorical_crossentropy(self):
        """when"""
        categories = 11
        loss = lossfunction.TailedCategoricalCrossentropyLoss(categories, 1)

        """then"""
        truth = one_hot(3, 11)
        l = loss(t.tensor([one_hot(6, 11)]), t.tensor([truth]))
        l2 = loss(t.tensor([one_hot(6, 11), one_hot(9, 11)]), t.tensor([truth, truth]))

        """then"""
        np.testing.assert_almost_equal(l, 11.817837, decimal=5)
        np.testing.assert_array_almost_equal(l2, [11.817837, 42.46247], decimal=5)
        self.assertGreater(l.mean().numpy(), 0)

    def test__tailed_categorical_crossentropy_3d(self):
        loss = lossfunction.TailedCategoricalCrossentropyLoss(11, 1)
        self.assertEqual(loss(t.ones((32, 11), requires_grad=True), t.ones((32, 1, 11), requires_grad=True)).shape, (32,))
        self.assertEqual(loss(t.ones((32, 1, 11), requires_grad=True), t.ones((32, 1, 11), requires_grad=True)).shape,  (32,))
        self.assertEqual(loss(t.ones((32, 11), requires_grad=True), t.ones((32, 11), requires_grad=True)).shape, (32,))

    def test_heteroscedasticity_loss_3d(self):
        from pandas_ml_utils_torch import lossfunction

        y_true = t.ones((10, 1, 3))
        y_pred = t.randn((10, 2, 3))
        critereon = lossfunction.HeteroscedasticityLoss(multi_nominal_reduction=None)

        t3d = critereon(y_pred, y_true)
        t2d = critereon(y_pred[:, :, 0], y_true[:, :, 0])

        print(t3d, '\n', t2d)
        np.testing.assert_array_equal(t2d.numpy(), t3d[:, 0].numpy())

