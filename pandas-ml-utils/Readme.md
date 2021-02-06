# Pandas ML Utils

Pandas Machine Learning Utilities is part of a bigger set of libraries for a convenient experience. Usually exploring
statistical models start with a pandas `DataFrame`. 

But soon enough you will find yourself converting your data frames to numpy, splitting arrays, applying min
max scalers, lagging and concatenating columns etc. As a result your notebook looks messy and became and 
unreadable beast. Yet the mess becomes only worse once you start to deploy your research into a productive
application. The untested hard coded data pipelines need be be maintained at two places. 

The aim of this library is to conveniently operate with data frames without and abstract away the ugly unreproducible 
data pipelines. The only thing you need is the original unprocessed data frame where you started.
The data pipeline becomes a part of your model and gets saved that way. Going into production is as easy as 
this:

```python

import pandas as pd
import pandas_ml_utils  # monkey patch the `DataFrame`
from pandas_ml_utils import Model
# alternatively as a one liner `from pandas_ml_utils import pd, Model` 

model = Model.load('your_saved.model')
df = pd.read_csv('your_raw_data.csv')
df_prediction = df.model.predict(model)

# do something with your prediction
df_prediction.plot()
``` 


is intended to help you through your journey of statistical or machine learning models, 
while you never need to leave the world of pandas.

## Installation
The basic implementation supports [scikit learn][e1] classifiers and regressors.
```shell script

pip install pandas-ml-utils
```

Additional machine learning libraries are available as an add on:
```shell script

pip install pandas-ml-utils-torch  # pytorch implementation
pip install pandas-ml-utils-keras  # keras + tensorflow 1.x implementation
```

Note that the keras/tensorflow version is currently stalled as I focus on pytorch recently. This might change
with PyMC4 and tensorflow probability
 
## Example
You will find some demo projects in the [examples][ghl1] directory. But It might also be worth it to check
the unit tests and the [integration tests][ghl2]. Here is how classification challenge
might look like:
  
![Classification Example][ghi1]


[e1]: https://scikit-learn.org/stable/
[ghl1]: ./examples/
[ghl2]: ../pandas-ml-1ntegration-test
[ghi1]: ../.readme/images/classification.png

