from pandas_ml_utils import FittingParameter
from pandas_ml_utils_test.ml.model.test_model_accuracy import TestModelAccuracy


class TestPytorchModelAccuracy(TestModelAccuracy):

    def provide_linear_regression_model(self):
        from pandas_ml_utils_torch import PytorchModel, PytorchNN
        from pandas_ml_utils import FeaturesAndLabels
        from torch.optim import Adam
        from torch import nn
        import torch as t

        class Net(PytorchNN):

            def __init__(self):
                super(Net, self).__init__()
                self.net = nn.Linear(1, 1)

            def forward_training(self, *input) -> t.Tensor:
                return self.net(input[0])

        return [
            (
                PytorchModel(Net, FeaturesAndLabels(["x"], ["y"]), nn.MSELoss, Adam),
                FittingParameter(epochs=5000, context="epoch fit"),
            ),
            (
                PytorchModel(Net, FeaturesAndLabels(["x"], ["y"]), nn.MSELoss, Adam),
                FittingParameter(epochs=5000, batch_size=64, context="epoch fit batched"),
            ),
            (
                PytorchModel(Net, FeaturesAndLabels(["x"], ["y"]), nn.MSELoss, Adam),
                FittingParameter(epochs=1, fold_epochs=5000, context="fold epoch fit"),
            ),
        ]

    def provide_non_linear_regression_model(self):
        from pandas_ml_utils_torch import PytorchModel, PytorchNN
        from pandas_ml_utils import FeaturesAndLabels
        from torch.optim import Adagrad
        from torch import nn
        import torch as t

        # t.manual_seed(0)

        class Net(PytorchNN):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 200),
                    nn.ReLU(),
                    nn.Linear(200, 200),
                    nn.ReLU(),
                    nn.Linear(200, 200),
                    nn.ReLU(),
                    nn.Linear(200, 1),
                    nn.ReLU()
                )

            def forward_training(self, *input) -> t.Tensor:
                return self.net(input[0])

        t.manual_seed(0)
        model = PytorchModel(Net, FeaturesAndLabels(["x"], ["y"]), nn.MSELoss, Adagrad)

        return [
            (
                model,
                FittingParameter(epochs=600, batch_size=64, context="epoch fit batched"),
            ),
            (
                model,
                FittingParameter(epochs=600, context="epoch fit"),
            ),
            (
                model,
                FittingParameter(epochs=1, fold_epochs=600, context="fold epoch fit"),
            )
        ]



