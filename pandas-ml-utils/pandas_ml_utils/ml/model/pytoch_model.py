from __future__ import annotations

import logging
import sys
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Type, Dict

import numpy as np

from pandas_ml_common import Typing
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
        self.history = {}

    def fit_fold(self,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> float:
        # import specifics
        from torch.autograd import Variable
        import torch as t

        restore_best_weights = kwargs["restore_best_weights"] if "restore_best_weights" in kwargs else False
        num_epochs = kwargs["epochs"] if "epochs" in kwargs else 100
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        use_cuda = kwargs["cuda"] if "cuda" in kwargs else False

        module = (self.module.cuda() if use_cuda else self.module).train()
        criterion = self.criterion_provider()
        optimizer = self.optimizer_provider(module.parameters())
        best_model_wts = deepcopy(module.state_dict())
        best_loss = sys.float_info.max
        epoch_losses = []
        epoch_val_losses = []

        for epoch in range(num_epochs):
            batch_loss = 0
            for i in range(0, len(x), batch_size):
                nnx = Variable(t.from_numpy(x[i:i+batch_size])).float()
                nny = Variable(t.from_numpy(y[i:i+batch_size])).float()
                weights = Variable(t.from_numpy(sample_weight_train[i:i+batch_size])).float() \
                    if sample_weight_train is not None else t.ones(len(x))

                if use_cuda:
                    nnx, nny, weights = nnx.cuda(), nny.cuda(), weights.cuda()

                # ===================forward=====================
                output = module(nnx)
                loss = (criterion(output, nny).sum() * weights).mean()

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            # ===================log========================
            # add loss history
            epoch_losses.append(batch_loss)

            # add validation loss history
            if y_val is not None and len(y_val) > 0:
                with t.no_grad():
                    nnx_val = Variable(t.from_numpy(x_val)).float()
                    nny_val = Variable(t.from_numpy(y_val)).float()

                    if use_cuda:
                        nnx_val, nny_val = nnx_val.cuda(), nny_val.cuda()

                    val_loss = self.criterion_provider()(module(nnx_val), nny_val).sum().item()
                    epoch_val_losses.append(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_wts = deepcopy(module.state_dict())

            # print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
            # print(output)
            if epoch % 10 == 0:
                #pic = to_img(output.cpu().data)
                #save_image(pic, './mlp_img/image_{}.png'.format(epoch))
                pass

        if restore_best_weights:
            module.load_state_dict(best_model_wts)

        self.history["loss"] = np.array(epoch_losses)
        self.history["val_loss"] = np.array(epoch_val_losses)

        return self.history["loss"][-1] if len(epoch_losses) > 0 else 0

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # import specifics
        from torch.autograd import Variable
        import torch as t

        use_cuda = kwargs["cuda"] if "cuda" in kwargs else False

        with t.no_grad():
            self.module.eval()

            if use_cuda:
                return self.module.cuda()(t.from_numpy(x).float().cuda()).numpy()
            else:
                return self.module(t.from_numpy(x).float()).numpy()

    def plot_loss(self):
        import matplotlib.pyplot as plt

        plt.plot(self.history['val_loss'], label='test')
        plt.plot(self.history['loss'], label='train')
        plt.legend(loc='best')

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes.
        # Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()

        # remove un-pickleable fields
        del state['module']

        # add torch serialisation
        state['module_state_dict'] = self.module.state_dict()

        # return altered state
        return state

    def __setstate__(self, state):
        # use torch.save(model.state_dict(), './sim_autoencoder.pth')
        # first remove the special state
        module_state_dict = state['module_state_dict']
        del state['module_state_dict']

        # Restore instance attributes
        self.__dict__.update(state)
        self.module = self.module_provider()

        # restore special state dict
        self.module.load_state_dict(module_state_dict)

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
