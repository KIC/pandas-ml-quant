from typing import Dict, Type, Callable, Tuple, Union, Optional, Generator

from ..preprocessing import FeaturesLabels, Extractor
from ..preprocessing.features_labels import FeaturesWithLabels
from ..sampling import Sampler, XYWeight
from ..typing import MlTypes


def sampling(df: MlTypes.PatchedDataFrame,
             features_and_labels_definition: FeaturesLabels,
             type_mapping: Optional[Dict[Type, callable]] = None,
             splitter: Optional[Callable[..., Tuple[MlTypes.PdIndex, MlTypes.PdIndex]]] = None,
             filter: Optional[Callable[..., bool]] = None,
             cross_validation: Optional[Union['BaseCrossValidator', Callable[..., Generator[Tuple[MlTypes.Array, MlTypes.Array], None, None]]]] = None,
             epochs: int = 1,
             batch_size: int = None,
             fold_epochs: int = 1,
             partition_row_multi_index_batch: bool = False,
             **kwargs) -> Tuple[FeaturesWithLabels, Sampler]:
    """
    This is all you need to get a training loop based on a single dataframe

    :param df: The dataframe with you features and labels
    :param features_and_labels_definition: The Definition which columns are features and which are labels
    :param type_mapping: Allows types to be mapped to callables i.e. for nesting models
    :param splitter: Splits the dataframe into training ad test frames
    :param filter:
    :param cross_validation:
    :param epochs: nr of epochs the generator will yield batches of the training data
    :param batch_size: size of the batches
    :param fold_epochs: nr of epochs the generator will loop batches of one cross-fold within the loop of overall epochs
    :param partition_row_multi_index_batch: in cases of timeseries or stationary data it may be necessary to split batches
        further into the level 0 parts of the multi index
    :param kwargs: keyword arguments to be passed to the extraction functions
    :return: returns a Sampler which yields batches of training data
    """

    extractor = Extractor(df, features_and_labels_definition, type_mapping=type_mapping, **kwargs)
    frames = extractor.extract_features_labels_weights()

    xyw = XYWeight(frames.features, frames.labels, frames.labels_with_sample_weights.sample_weights)

    # set up a sampler for the data
    return frames, Sampler(
        xyw,
        splitter=splitter,
        filter=filter,
        cross_validation=cross_validation,
        epochs=epochs,
        fold_epochs=fold_epochs,
        batch_size=batch_size,
        partition_row_multi_index=partition_row_multi_index_batch
    )
