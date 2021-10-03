## The pandas ml common module

This module holds all necessary extensions to get from a single `DataFrame` to a state-of-the-art training loop. 
There are also plenty of utils for easy access of data stored in pandas `DataFrame`s like nested lists or easy `getter` 
functions for `MultiIndex` frames and more. Feel free to study the [examples][ghl1] as well.

* easy joining of data frames with multi indexes
```python
from pandas_ml_common import pd, np

df1 = pd.DataFrame({"a": np.random.random(10), "b": np.random.random(10)})
print(df1.inner_join(df1, prefix_left='A', prefix='B', force_multi_index=True).to_markdown())
```
|    |   ('A', 'a') |   ('A', 'b') |   ('B', 'a') |   ('B', 'b') |
|---:|-------------:|-------------:|-------------:|-------------:|
|  0 |     0.907892 |    0.726913  |     0.907892 |    0.726913  |
|  1 |     0.602275 |    0.134278  |     0.602275 |    0.134278  |
|  2 |     0.264399 |    0.207429  |     0.264399 |    0.207429  |
|  3 |     0.559751 |    0.816759  |     0.559751 |    0.816759  |
|  4 |     0.951172 |    0.797524  |     0.951172 |    0.797524  |
|  5 |     0.504332 |    0.51996   |     0.504332 |    0.51996   |
|  6 |     0.765235 |    0.17908   |     0.765235 |    0.17908   |
|  7 |     0.388691 |    0.644103  |     0.388691 |    0.644103  |
|  8 |     0.663636 |    0.678879  |     0.663636 |    0.678879  |
|  9 |     0.291603 |    0.0164627 |     0.291603 |    0.0164627 |


* access columns with regex
```python
df4 = pd.DataFrame({"a_22_a": np.random.random(1), "b_21_b": np.random.random(1)})
df4.ML[r'.*\d+_.']
```
|    |   a_22_a |    b_21_b |
|---:|---------:|----------:|
|  0 |  0.22039 | 0.0374084 |



* easy access multi level index
```python
df1.unique_level_columns(0)

['A', 'B']

df1.add_multi_index('Z', axis=1)
```

* data splitting, sampling and folding (aka cross validation)
```python
from pandas_ml_common import Sampler, XYWeight, random_splitter

df2 = pd.DataFrame({"c": np.random.random(10)})
sampler = Sampler(XYWeight(df1, df2), splitter=random_splitter(0.5))

for batches in sampler.sample_for_training():
    for batch in batches:
        print(batch)
```


* access to nested numpy arrays in data frame columns (`df.ML.values`)
```python
df3 = pd.DataFrame({"a": [[1, 2], [3, 4], [5, 6]]})
df3.ML.values

array([[1, 2],
       [3, 4],
       [5, 6]])
```


* dynamic method call providing suitable *args and **kwargs (dependency injection)
```python
from pandas_ml_common import call_callable_dynamic_args

def adder(a, b):
    return a + b

call_callable_dynamic_args(adder, a=12, b=10, c='illegal')

22
```


* numpy utils 
```python

from pandas_ml_common import np_nans

np_nans((3, 3))

array([[nan, nan, nan],
       [nan, nan, nan],
       [nan, nan, nan]])


from pandas_ml_common import temp_seed

with temp_seed(42):
    print(np.random.random(2))

np.random.random(2)


[0.37454012 0.95071431]
array([0.69373278, 0.69790163])
```


* serialization utils
```python
from pandas_ml_common import serializeb, deserializeb

deserializeb(serializeb(np.array([1, 2, 3])))
array([1, 2, 3])
```

* re-scalings

```python
from pandas_ml_common import ReScaler

x = np.arange(0, 1, .1)
rescaler = ReScaler((0, 1), (5, -5))

rescaler(x)
array([ 5.,  4.,  3.,  2.,  1.,  0., -1., -2., -3., -4.])
```

[ghl1]: ./examples/