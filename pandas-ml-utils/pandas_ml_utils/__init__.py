"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.0'

from pandas_ml_common import *
from pandas_ml_utils.data.extraction import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")

