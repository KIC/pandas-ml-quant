"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.2.7'

from pandas_ml_utils.ml.summary import *
from pandas_ml_common import *
from pandas_ml_utils.df_patching.model_patch import DfModelPatch
from pandas_ml_utils.ml.data import *
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels, PostProcessedFeaturesAndLabels
from pandas_ml_utils.ml.model import *
from pandas_ml_utils.ml.fitting import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "model", property(lambda self: DfModelPatch(self)))
