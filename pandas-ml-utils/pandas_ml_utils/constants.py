# import privately to allow star import
import numpy as _np

PREDICTION_COLUMN_NAME = "prediction"
FEATURE_COLUMN_NAME = "feature"
TARGET_COLUMN_NAME = "target"
LABEL_COLUMN_NAME = "label"
GROSS_LOSS_COLUMN_NAME = "gross_loss"
SOURCE_COLUMN_NAME = "source"
SAMPLE_WEIGHTS_COLUMN_NAME = "sample_weights"

PROBABILITY_POSTFIX = "_proba"

SIMULATED_VECTOR = _np.ones(10000)