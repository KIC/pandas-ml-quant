# Pandas TA Quant

Not only a pure python re-implementation of the famous [TA-Lib][e1]. Additional indicators are available like covariance 
measures or arma, garch and sarimax models. The library fully builds on top of pandas and pandas_ml_common, therefore
allows to deal with MultiIndex easily:

| Date                |   ('spy', 'Open') |   ('spy', 'High') |   ('spy', 'Low') |   ('spy', 'Close') |   ('spy', 'Volume') |   ('spy', 'Dividends') |   ('spy', 'Stock Splits') |   ('gld', 'Open') |   ('gld', 'High') |   ('gld', 'Low') |   ('gld', 'Close') |   ('gld', 'Volume') |   ('gld', 'Dividends') |   ('gld', 'Stock Splits') |
|:--------------------|------------------:|------------------:|-----------------:|-------------------:|--------------------:|-----------------------:|--------------------------:|------------------:|------------------:|-----------------:|-------------------:|--------------------:|-----------------------:|--------------------------:|
| 2020-02-07 00:00:00 |            332.82 |            333.99 |           331.6  |             332.2  |         6.41394e+07 |                      0 |                         0 |            147.83 |            148.18 |           147.34 |             147.79 |         6.3793e+06  |                      0 |                         0 |
| 2020-02-10 00:00:00 |            331.23 |            334.75 |           331.19 |             334.68 |         4.207e+07   |                      0 |                         0 |            148.21 |            148.45 |           147.91 |             148.17 |         5.7936e+06  |                      0 |                         0 |

```
df = pd.read_pickle("../pandas_ta_quant_test/.data/spy_gld.pickle")
df._[["Close", df._["Close"].ta.sma(200)]].plot(figsize=(20,10))
```

![Plot][ghi1]

## Full List of indicators

|                                | module                                                            |
|:-------------------------------|:------------------------------------------------------------------|
| ta_adx                         | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_all                         | pandas_ta_quant.technical_analysis.indicators                     |
| ta_apo                         | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_atr                         | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_bbands                      | pandas_ta_quant.technical_analysis.bands                          |
| ta_bbands_indicator            | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_bop                         | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_candle_category             | pandas_ta_quant.technical_analysis.encoders.candles               |
| ta_candles_as_culb             | pandas_ta_quant.technical_analysis.encoders.candles               |
| ta_cci                         | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_cross                       | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_cross_over                  | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_cross_under                 | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_decimal_year                | pandas_ta_quant.technical_analysis.indicators.time                |
| ta_delta_hedged_price          | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_div                         | pandas_ta_quant.technical_analysis.math                           |
| ta_draw_down                   | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_edge_detect                 | pandas_ta_quant.technical_analysis.forecast.support               |
| ta_ema                         | pandas_ta_quant.technical_analysis.filters                        |
| ta_ewma_covariance             | pandas_ta_quant.technical_analysis.covariances                    |
| ta_fibbonaci_retracement       | pandas_ta_quant.technical_analysis.forecast.support               |
| ta_future_bband_quantile       | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_future_crossings            | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_future_multi_bband_quantile | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_future_multi_ma_quantile    | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_future_pct_to_current_mean  | pandas_ta_quant.technical_analysis.labels.continuous              |
| ta_gaf                         | pandas_ta_quant.technical_analysis.encoders.gramian_angular_field |
| ta_gap                         | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_garch11                     | pandas_ta_quant.technical_analysis.forecast.volatility            |
| ta_has_opening_gap             | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_hmm                         | pandas_ta_quant.technical_analysis.forecast.predictive_indicator  |
| ta_inverse                     | pandas_ta_quant.technical_analysis.encoders.resample              |
| ta_inverse_gasf                | pandas_ta_quant.technical_analysis.encoders.gramian_angular_field |
| ta_is_opening_gap_closed       | pandas_ta_quant.technical_analysis.labels.discrete                |
| ta_log_returns                 | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_ma_decompose                | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_ma_ratio                    | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_macd                        | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_mean_returns                | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_mgarch_covariance           | pandas_ta_quant.technical_analysis.covariances                    |
| ta_mom                         | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_moving_covariance           | pandas_ta_quant.technical_analysis.covariances                    |
| ta_multi_bbands                | pandas_ta_quant.technical_analysis.filters                        |
| ta_multi_ma                    | pandas_ta_quant.technical_analysis.filters                        |
| ta_ncdf_compress               | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_normalize_row               | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_ohl_trend_lines             | pandas_ta_quant.technical_analysis.forecast.support               |
| ta_one_hot                     | pandas_ta_quant.technical_analysis.encoders.one_hot               |
| ta_one_hot_encode_discrete     | pandas_ta_quant.technical_analysis.encoders.one_hot               |
| ta_performance                 | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_poly_coeff                  | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_ppo                         | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_realative_candles           | pandas_ta_quant.technical_analysis.encoders.candles               |
| ta_rescale                     | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_returns                     | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_rnn                         | pandas_ta_quant.technical_analysis.encoders.auto_regression       |
| ta_roc                         | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_rsi                         | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_sarimax                     | pandas_ta_quant.technical_analysis.forecast.predictive_indicator  |
| ta_sharpe_ratio                | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_sinusoidal_week             | pandas_ta_quant.technical_analysis.indicators.time                |
| ta_sinusoidal_week_day         | pandas_ta_quant.technical_analysis.indicators.time                |
| ta_slope                       | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_sma                         | pandas_ta_quant.technical_analysis.filters                        |
| ta_sma_price_ratio             | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_sortino_ratio               | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_sparse_covariance           | pandas_ta_quant.technical_analysis.covariances                    |
| ta_std_ret_bands               | pandas_ta_quant.technical_analysis.bands                          |
| ta_std_ret_bands_indicator     | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_stddev                      | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_tr                          | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_trend_lines                 | pandas_ta_quant.technical_analysis.forecast.support               |
| ta_trix                        | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_ultimate_osc                | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_up_down_volatility_ratio    | pandas_ta_quant.technical_analysis.indicators.single_object       |
| ta_volume_as_time              | pandas_ta_quant.technical_analysis.encoders.volume                |
| ta_wilders                     | pandas_ta_quant.technical_analysis.filters                        |
| ta_williams_R                  | pandas_ta_quant.technical_analysis.indicators.multi_object        |
| ta_z_norm                      | pandas_ta_quant.technical_analysis.normalizer                     |
| ta_zscore                      | pandas_ta_quant.technical_analysis.indicators.single_object       |                                                                       |

[ghi1]: ../.readme/images/multi_index.png

[e1]: http://mrjbq7.github.io/ta-lib/