import os
import unittest
from unittest import TestCase

from pandas_ml_utils import pd, Model
from pandas_ml_common_test.notebook_runner import run_notebook

PWD = os.path.dirname(os.path.abspath(__file__))


class TestSaveAndLoad(TestCase):

    def test_save_load_nb_model(self):
        notebooks_path = os.path.join(PWD, '..', 'examples')
        notebook_file = os.path.join(notebooks_path, 'regression_with_regularization.ipynb')
        out, err = run_notebook(notebook_file, notebooks_path)
        self.assertEqual(err, [])

        df = pd.read_csv(os.path.join(notebooks_path, 'SPY.csv'))
        model = Model.load('/tmp/regression_with_regularization.model')
        prediction = df.model.predict(model, tail=1)["prediction", "Close"].item()

        print(prediction)
        self.assertEqual(float(out["cells"][-1]["outputs"][-1]["data"]["text/plain"]), prediction)


    @unittest.skip("debuging")
    def test_make_model(self):
        notebooks_path = os.path.join(PWD, '..', 'examples')
        df = pd.read_csv(os.path.join(notebooks_path, 'SPY.csv'))

        with df.model("/tmp/pijsfnwuacpa.model") as m:
            from torch import nn
            from torch.optim import SGD
            from pandas_ml_common.utils.column_lagging_utils import lag_columns

            from pandas_ml_utils import FeaturesAndLabels, RegressionSummary, FittingParameter
            from pandas_ml_utils_torch import PytorchModel
            from pandas_ml_utils_torch.merging_cross_folds import take_the_best

            def net_provider():
                from pandas_ml_utils_torch import PytorchNN

                class Net(PytorchNN):

                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(10, 4),
                            nn.Tanh(),
                            nn.Linear(4, 4),
                            nn.Tanh(),
                            nn.Linear(4, 1),
                            nn.Tanh(),
                        )

                    def L1(self):
                        # path to the parameters which should be regularized
                        # the path is constructed from self.named_parameters() and allows the use of wildcards
                        return {'net/0/**/weight': 0.02}

                    def L2(self):
                        return {
                            'net/0/**/weight': 0.02,
                            'net/2/**/weight': 0.05
                        }

                    def forward_training(self, x):
                        return self.net(x)

                return Net()

            fit = m.fit(
                PytorchModel(
                    net_provider,
                    FeaturesAndLabels(
                        [lambda df: lag_columns(df["Close"].pct_change(), range(10))],
                        [lambda df: df["Close"].pct_change().shift(-1)]),
                    nn.MSELoss,
                    lambda params: SGD(params, lr=0.01, momentum=0.0),
                    merge_cross_folds=take_the_best,
                    summary_provider=RegressionSummary
                ),
                FittingParameter(epochs=2),
                verbose=1
            )

    @unittest.skip("debuging")
    def test_load_only(self):
        from pandas_ml_common.utils.column_lagging_utils import lag_columns
        notebooks_path = os.path.join(PWD, '..', 'examples')
        df = pd.read_csv(os.path.join(notebooks_path, 'SPY.csv'))
        #model = Model.load("/tmp/pijsfnwuacpa.model")
        model = Model.load('/tmp/regression_with_regularization.model')
        prediction = df.model.predict(model, tail=1)

        print(prediction)
        self.assertEqual(len(prediction), 1)
