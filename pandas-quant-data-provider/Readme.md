# Pandas ML Quant Data Provider

An independent module used to fetch time series and other data used for quant finance studies.

## Installation

Data can always be fetched instant without any local data base. However if you need to screen across assets
for matching criteria then a local database is needed. Developing models often depend on some premise. Like 
for a crash detection model you would want to use instruments which indeed suffered a crash. Therefore some
data is stored into a data base using dolthub. Querying this data requires you to install [dolt][dolt] by
following their [installation instructions][dolt].  

```shell script
pip install pandas-quant-data-provider
 
```
   
**!! NOTE !!**
We need to keep data libraries like yfincance up do date alsmost every minute. Whenever there is a change in the api
the library has to react. This happens more often as we whish, especially with yahoo finance. This is why this library
gets _pip installed -U_ on every import. This means for server based solutions on every restart. Make sure you use
all pandas-ml* libraries inside a virtual env.


 
Example:

```python
# monkey patch pandas 
from pandas_quant_data_provider import pd, YAHOO, INVESTING, CRYPTO_COMPARE

# fetch data from various data sources 
#   * fetches all available dates
#   * caches data for 10 minutes
df = pd.fetch_timeseries({
    YAHOO: ["SPY", "DIA"],
    INVESTING: ["index::NYSE Tick Index::united states", "bond::U.S. 30Y::united states"],
    CRYPTO_COMPARE: ["BTC"]
})

df.tail()
```

PS If you are not familiar with pandas MultiIndex, you can watch this video:
[How do I use the MultiIndex in pandas?](https://www.youtube.com/watch?v=tcRGa2soc-c)

 
## Installation
Follow the instructions on [https://github.com/KIC/pandas-ml-quant](https://github.com/KIC/pandas-ml-quant)

## Documentation
Check out the notebooks at [https://github.com/KIC/pandas-ml-quant/blob/master/notebooks](https://github.com/KIC/pandas-ml-quant/blob/master/notebooks)


[dolt]: https://www.dolthub.com/blog/2020-02-03-dolt-and-dolthub-getting-started/