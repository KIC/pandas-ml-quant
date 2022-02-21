from .encoder import ta_one_hot_encode_discrete as _ta_one_hot_encode_discrete
from .labels.discrete import ta_onehot_idx
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
#                                              A L I A S E S                                                           #
# -------------------------------------------------------------------------------------------------------------------- #

ta_one_hot = _ta_one_hot_encode_discrete
ta_one_hot_argmax = lambda df, *args, **kwargs: ta_onehot_idx(df, *args, **kwargs, func=np.argmax)
