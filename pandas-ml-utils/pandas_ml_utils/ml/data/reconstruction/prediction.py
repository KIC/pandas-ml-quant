from typing import Dict

# make sure to import monkey patched data frame
from pandas_ml_common import pd, np, get_pandas_object, add_multi_index
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME
from pandas_ml_utils.constants import *


def assemble_result_frame(prediction, targets, labels, gross_loss, weights, features):
    return assemble_prediction_frame(
        {PREDICTION_COLUMN_NAME: prediction,
         TARGET_COLUMN_NAME: targets,
         LABEL_COLUMN_NAME: labels,
         GROSS_LOSS_COLUMN_NAME: gross_loss,
         SAMPLE_WEIGHTS_COLUMN_NAME: weights,
         FEATURE_COLUMN_NAME: features})


def assemble_prediction_frame(frames: Dict[str, pd.DataFrame]):
    # filter non frames
    valid_frames = {head: (frame.as_joined_frame() if isinstance(frame, MultiFrameDecorator) else frame.copy())
                    for head, frame in frames.items() if frame is not None }

    # convert into MultiIndex column headers
    for head, frame in valid_frames.items():
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
            valid_frames[head] = frame

        frame.columns = pd.MultiIndex.from_product([[head], frame.columns.to_list()])

    # join all frames and keep the order of the passed dictionary
    df = pd.concat(valid_frames.values(), axis=1, join='inner', copy=False)

    # monkey patch prediction frame
    df.map_prediction_to_target = lambda: map_prediction_to_target(df, PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME)
    return df


def map_prediction_to_target(df, prediction, targets):
    def _round(val, d):
        return round(val, d) if isinstance(val, float) else val

    dfp = get_pandas_object(df, prediction)
    p = dfp._.values.reshape((len(df), -1))

    dft = get_pandas_object(df, targets)
    t = dft._.values.reshape((len(df), -1))

    if p.shape[1] == t.shape[1]:
        # 1:1 mapping
        index = [(date, _round(target, 2)) for date in df.index for target in dft.loc[date].values]
    elif p.shape[1] == t.shape[1] - 1:
        # we need to build ranges
        def build_tuples(l):
            return [(_round(l[i - 1], 2), _round(l[i], 2)) for i in range(1, len(l))]

        index = [(date, f"{target}") for date in df.index for target in
                 build_tuples(dft.loc[date].tolist())]
    elif p.shape[1] == t.shape[1] + 1:
        # mapping of the left and right extremes using +/- inf
        def build_tuples(l):
            l = [-np.inf, *l, np.inf]
            return [(_round(l[i - 1], 2), _round(l[i], 2)) for i in range(1, len(l))]

        index = [(date, target) for date in df.index for target in
                 build_tuples(dft.loc[date]._.values.tolist())]
    else:
        raise ValueError(f"unable to match {p.shape[1]} predictions to {t.shape[1]} +/-1 targets")

    return pd.DataFrame({"prediction": p.reshape((-1,))},
                        index=pd.MultiIndex.from_tuples(index))

