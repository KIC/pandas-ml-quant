from typing import Dict

import pandas as pd


def assemble_prediction_frame(frames: Dict[str, pd.DataFrame]):
    # filter non frames
    valid_frames = {head: frame for head, frame in frames.items() if frame is not None}

    for head, frame in valid_frames.items():
        frame.columns = pd.MultiIndex.from_product([[head], frame.columns.to_list()])

    # join all frames and keep the order of the passed dictionary
    return pd.concat(valid_frames.values(), axis=0, copy=False)[valid_frames.keys()]


"""
def prediction_to_frame(df,
                        prediction: np.ndarray,
                        index: pd.Index = None,
                        inclusive_labels: bool = False,
                        inclusive_source: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # sanity check
    if not isinstance(prediction, np.ndarray):
        raise ValueError(f"got unexpected prediction: {type(prediction)}\n{prediction}")

    # assign index
    index = self._df.index if index is None else index

    # eventually fix the shape of the prediction
    if len(prediction.shape) == 1:
        prediction = prediction.reshape(len(prediction), 1)

    # prediction_columns
    columns = pd.MultiIndex.from_tuples(self.label_names(PREDICTION_COLUMN_NAME))
    multi_dimension_prediction = len(prediction.shape) > 1 and len(columns) < prediction.shape[1]
    if multi_dimension_prediction:
        if len(prediction.shape) < 3:
            df = pd.DataFrame({"a": [r.tolist() for r in prediction]}, index=index)
        else:
            df = pd.DataFrame({col: [row.tolist() for row in prediction[:, col]] for col in range(prediction.shape[1])},
                              index=index)

        df.columns = columns
    else:
        df = pd.DataFrame(prediction, index=index, columns=columns)

    # add labels if requested
    if inclusive_labels:
        dfl = self.labels_df
        dfl.columns = pd.MultiIndex.from_tuples(self.label_names(LABEL_COLUMN_NAME))
        df = df.join(dfl, how='inner')

        # add loss if provided
        loss_df = self.gross_loss_df
        df = df.join(loss_df.loc[df.index], how='inner') if loss_df is not None else df

    # add target if provided
    target_df = self.target_df
    df = df.join(target_df.loc[df.index], how='inner') if target_df is not None else df

    # also add source if requested
    if inclusive_source:
        df = df.join(self.source_df, how='inner')

    # finally we can return our nice and shiny df
    return df
"""
