from typing import Union as _Union
from pyts.image import GramianAngularField as _GAF
import pandas as _pd
_PANDAS = _Union[_pd.DataFrame, _pd.Series]


# FIXME
def ta_gaf(df: _pd.DataFrame,
          columm_index_level=1,
          image_size=24,
          sample_range=(-1, 1),
          method='summation',
          flatten=False,
          overlapping=False):

    if isinstance(df.columns, _pd.MultiIndex):
        # for each n'd level column
        l = columm_index_level
        columns = {c[l]: [c2 for c2 in df.columns.to_list() if c2[l] == c[l]] for c in df.columns.to_list()}
        res = _pd.DataFrame({}, index=df.index)

        for feature, timesteps in columns.items():
            dff = df[timesteps]
            dff.columns = [t[:l] + t[l+1:] for t in timesteps]
            s = ta_gaf(dff, columm_index_level, image_size, sample_range, method, flatten, overlapping)
            res[f'{feature}_gaf'] = s

        return res
    else:
        image_size = max(min(image_size, len(df.columns)), 1)
        gaf = _GAF(image_size=image_size, sample_range=sample_range, method=method,
                   overlapping=overlapping, flatten=flatten)

        return _pd.Series(list(gaf.fit_transform(df.values)), name="gaf", index=df.index)


