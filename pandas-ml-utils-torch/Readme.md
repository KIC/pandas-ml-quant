# pytorch extension for [pandas-ml-utils][ghl1]

Adds a `PytorchModel` to the pandas ml utils suite. While a regular class extending `nn.Module` is sufficient,
there is also a special class `PytorchNN` which can be extended as well. Using this class has the following 
advantages:

 * allows to use L1, L2 regularisation 
 * allows different forward path for training and prediction (useful i.e. for re-parametrisation trick)
 * allows to implement auto-encoders easily by just providing the encode/decode functions

See also this [example][ghl2].

[ghl1]: ../pandas-ml-utils
[ghl2]: ./examples/regression_with_regularization.ipynb


