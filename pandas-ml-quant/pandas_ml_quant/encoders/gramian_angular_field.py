# FIXME

"""
def ta_gaf(df: _PANDAS,
          period=20,
          image_size=24,
          sample_range=(-1, 1),
          method='summation',
          flatten=False,
          overlapping=False):

    def to_gaf(df):
        gaf = GramianAngularField(image_size=image_size, sample_range=sample_range, method=method,
                                  overlapping=overlapping, flatten=flatten)
        return gaf.fit_transform(df.values)

    return _pd.Series([to_gaf(df.iloc[i-period, period]) for i in range(period, len(df))], index=df.index, name="GAF")

"""