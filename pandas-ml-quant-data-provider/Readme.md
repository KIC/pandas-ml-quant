# Pandas ML Quant Data Provider

An independent module used to fetch time series data for quant finance studies.
 
Example:
```python
# monkey patch pandas 
from pandas_ml_quant_data_provider import pd, YAHOO, INVESTING, CRYPTO_COMPARE

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

