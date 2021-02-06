# pytorch extension for [pandas-ml-utils][ghl1]

Adds a `PytorchModel` to the pandas ml utils suite. While a regular class extending `nn.Module` is sufficient,
there is also a special class `PytorchNN` which can be extended as well. Using this class has the following 
advantages:

 * allows to use L1, L2 regularization -> [example][ghl2] 
 * allows different forward path for training and prediction (useful i.e. for reparameterization trick) -> [example][ghl3]
 * allows to implement auto-encoders easily by just providing the encode/decode functions
 * added loss functions like `SoftDTW` (fit time series) loss or `HeteroscedasticityLoss` (fit Normal Distribution) -> [example][ghl3]

<br><br>

![Fitting Example][ghi1]

[ghl1]: ../pandas-ml-utils
[ghl2]: ./examples/regression_with_regularization.ipynb
[ghl3]: ./examples/probabilistic_model.ipynb
[ghi1]: ../.readme/videos/probabilistic-model-fit.gif

