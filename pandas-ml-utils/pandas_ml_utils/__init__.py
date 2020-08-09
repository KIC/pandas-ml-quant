"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.14.1'

from pandas_ml_utils.ml.summary import *
from pandas_ml_common import *
from pandas_ml_utils.df.model import Model as DfModelExtension
from pandas_ml_utils.ml.data import *
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.model import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "model", property(lambda self: DfModelExtension(self)))
