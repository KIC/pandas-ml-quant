from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import uuid
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Tuple, Type, Dict

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import merge_kwargs, suitable_kwargs
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class PytorchModel(Model):

    if TYPE_CHECKING:
        from torch.nn import Module, _Loss
        from torch.optim import Optimizer

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 module_provider: Type[Module],
                 criterion_provider: Type[_Loss],
                 optimizer_provider: Type[Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 callbacks: Dict[str, List[Callable]] = {},
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.module_provider = module_provider
        self.criterion_provider = criterion_provider
        self.optimizer_provider = optimizer_provider
        self.callbacks = callbacks
        self.module = None

    def fit_fold(self,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> float:
        # import specifics
        from torch.autograd import Variable
        import torch as t

        num_epochs = kwargs["epochs"] if "epochs" in kwargs else 100
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        use_cuda = kwargs["cuda"] if "cuda" in kwargs else False

        module = (self.module.cuda() if use_cuda else self.module).train()
        criterion = self.criterion_provider()
        optimizer = self.optimizer_provider(module.parameters())

        for epoch in range(num_epochs):
            for i in range(0, len(x), batch_size):
                nnx = Variable(t.from_numpy(x[i:i+batch_size])).float()
                nny = Variable(t.from_numpy(y[i:i+batch_size])).float()

                if use_cuda:
                    nnx, nny = nnx.cuda(), nny.cuda()

                # ===================forward=====================
                output = module(nnx)
                loss = criterion(output, nny)

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data))
            # print(output)
            if epoch % 10 == 0:
                #pic = to_img(output.cpu().data)
                #save_image(pic, './mlp_img/image_{}.png'.format(epoch))
                pass

        return 0 # fixme return loss

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # import specifics
        from torch.autograd import Variable
        import torch as t

        use_cuda = kwargs["cuda"] if "cuda" in kwargs else False

        with t.no_grad():
            self.module.eval()

            if use_cuda:
                return self.module.cuda()(Variable(t.from_numpy(x)).float().cuda()).numpy()
            else:
                return self.module(Variable(t.from_numpy(x)).float()).numpy()

    # TODO serialization
    #  def __getstate__(self):
    #  def __setstate__(self): use torch.save(model.state_dict(), './sim_autoencoder.pth')

    def __call__(self, *args, **kwargs):
        pytorch_model = PytorchModel(
            self.features_and_labels,
            self.module_provider,
            self.criterion_provider,
            self.optimizer_provider,
            self.summary_provider,
            deepcopy(self.callbacks)
        )

        pytorch_model.module = self.module_provider()
        return pytorch_model
