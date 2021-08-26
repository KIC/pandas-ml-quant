import pandas as pd


def pd_concat(frames: pd.DataFrame, default=None, *args, **kwargs):
    return pd.concat(frames, *args, **kwargs) if len(frames) > 0 else default
