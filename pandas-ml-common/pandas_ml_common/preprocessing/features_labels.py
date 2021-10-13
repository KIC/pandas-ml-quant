import logging
from typing import NamedTuple, List, TypeVar, Type, Optional, Dict, Tuple, Set, Union, Iterable, Callable

import numpy as np

from ..typing import MlTypes
from ..utils import call_if_not_none, get_pandas_object, intersection_of_index, loc_if_not_none, is_in_index, \
    make_same_length, call_callable_dynamic_args, none_as_empty_list, flatten_nested_list, none_as_empty_dict, GetItem, \
    pd_concat, pd_dropna

_log = logging.getLogger(__name__)

IndexLookup = Union[List[MlTypes.DataSelector], Set[MlTypes.DataSelector], Tuple[MlTypes.DataSelector]]
NestedIndexLookup = Union[List[IndexLookup], Set[IndexLookup], Tuple[IndexLookup]]
FeatureSelectors = TypeVar('FeatureSelectors', MlTypes.DataSelector, IndexLookup, NestedIndexLookup)
LabelSelectors = TypeVar('LabelSelectors', MlTypes.DataSelector, IndexLookup, NestedIndexLookup)
PostProcess = TypeVar('PostProcess', MlTypes.DataSelector, IndexLookup, NestedIndexLookup)
LabelType = TypeVar('LabelType', Type[float], Type[int], List[Union[None, Type[float], Type[int]]])


class FeaturesLabels(NamedTuple):
    features: FeatureSelectors
    features_postprocessor: Optional[PostProcess] = None
    labels: LabelSelectors = ()
    labels_postprocessor: Optional[PostProcess] = None
    sample_weights: Optional[MlTypes.DataSelector] = None
    sample_weights_postprocessor: Optional[PostProcess] = None
    gross_loss: Optional[MlTypes.DataSelector] = None
    gross_loss_postprocessor: Optional[PostProcess] = None
    reconstruction_targets: Optional[MlTypes.DataSelector] = None
    reconstruction_targets_postprocessor: Optional[PostProcess] = None
    label_type: Optional[LabelType] = None


class FeaturesWithReconstructionTargets(NamedTuple):
    features: List[MlTypes.PatchedDataFrame]
    reconstruction_targets: Optional[MlTypes.PatchedDataFrame]
    min_required_samples: int

    def with_features(self, new_features: List[MlTypes.PatchedDataFrame]):
        assert [nf.shape for nf in new_features] == [f.shape for f in self.features], "Incompatible Shapes!"
        return FeaturesWithReconstructionTargets(
            new_features,
            self.reconstruction_targets,
            self.min_required_samples,
        )

    @property
    def joint_feature_frame(self):
        return pd_concat(self.features, multiindex_columns=True, dedupe_columns=True)

    @property
    def common_index(self):
        return intersection_of_index(*self.features)

    @property
    def loc(self) -> GetItem['FeaturesWithReconstructionTargets']:
        return GetItem(
            lambda idx: FeaturesWithReconstructionTargets(
                *[loc_if_not_none(f, idx) for f in [self.features, self.reconstruction_targets]],
                min_required_samples = self.min_required_samples
            )
        )

    @property
    def shape(self) -> Dict:
        return {
            'features': [(len(f),) + f[:1].ML.values.shape for f in self.features],
            'reconstruction': (len(self.reconstruction_targets),) + self.reconstruction_targets[:1].ML.values.shape if self.reconstruction_targets is not None else (),
            'min_required_samples': self.min_required_samples,
        }

    def __len__(self):
        return len(self.features)


class LabelsWithSampleWeights(NamedTuple):
    labels: List[MlTypes.PatchedDataFrame]
    sample_weights: Optional[List[Optional[MlTypes.PatchedDataFrame]]]
    gross_loss: Optional[MlTypes.PatchedDataFrame] = None

    @property
    def joint_label_frame(self):
        return pd_concat(self.labels)

    @property
    def joint_sample_weights_frame(self):
        return pd_concat(self.sample_weights)

    @property
    def shape(self) -> Dict:
        return {
            'labels': [(len(l),) + l[:1].ML.values.shape for l in self.labels],
            'sample_weights': [((len(sw),) + sw[:1].ML.values.shape if sw is not None else None) for sw in self.sample_weights],
            'gross_loss': (len(self.gross_loss),) + self.gross_loss[:1].ML.values.shape if self.gross_loss is not None else (),
        }


class FeaturesWithLabels(NamedTuple):
    features_with_required_samples: FeaturesWithReconstructionTargets
    labels_with_sample_weights: LabelsWithSampleWeights

    @property
    def common_index(self):
        return intersection_of_index(self.common_features_index, *self.labels)

    @property
    def common_features_index(self):
        return intersection_of_index(*self.features)

    @property
    def features(self) -> List[MlTypes.PatchedDataFrame]:
        return self.features_with_required_samples.features

    @property
    def labels(self) -> List[MlTypes.PatchedDataFrame]:
        return self.labels_with_sample_weights.labels

    @property
    def shape(self) -> Dict[str, Dict]:
        return {
            'features': self.features_with_required_samples.shape,
            'labels': self.labels_with_sample_weights.shape,
        }


class Extractor(object):

    def __init__(self,
                 df: MlTypes.PatchedDataFrame,
                 features_labels: FeaturesLabels,
                 type_mapping: Dict[Type, Callable[[MlTypes.DataFrame], MlTypes.DataFrame]] = None,
                 **kwargs):
        self.df = df
        self.features_and_labels_definition = features_labels
        self.type_map = none_as_empty_dict(type_mapping)
        self.kwargs = kwargs

    def extract_features_labels_weights(self, tail=None) -> FeaturesWithLabels:
        features = self.extract_features(tail=tail)
        labels = self.extract_labels()

        # do some sanity check for any non-numeric values in any of the data frames
        all_frames: List[MlTypes.PatchedDataFrame] = [
            *features.features, *none_as_empty_list(features.reconstruction_targets),
            *labels.labels, *none_as_empty_list(labels.sample_weights), *none_as_empty_list(labels.gross_loss)
        ]

        for frame in all_frames:
            values = flatten_nested_list(frame.ML.values, np.max)
            max_value = max([v.max() for v in values])

            if np.isscalar(max_value) and np.isinf(max_value):
                _log.warning(f"frames containing infinit numbers\n"
                             f"{frame[frame.apply(lambda r: np.isinf(r.values).any(), axis=1)]}")
                frame.replace([np.inf, -np.inf], np.nan, inplace=True)
                frame.dropna(inplace=True)

        return FeaturesWithLabels(features, labels)

    def extract_features(self, tail=None) -> FeaturesWithReconstructionTargets:
        # extract 2D features
        features = self._extract_eventaually_nested_selectors(
            self.df if tail is None else self.df.tail(abs(tail)),
            self.features_and_labels_definition.features
        )

        # however one model can only have one frame of reconstruction targets
        reconstruction_targets = self.features_and_labels_definition.reconstruction_targets
        recon_tgt = call_if_not_none(get_pandas_object(self.df, reconstruction_targets, type_map=self.type_map, **self.kwargs), 'dropna')

        # execute post-processors on features and targets !!!
        features = self._apply_post_processor(features, self.features_and_labels_definition.features_postprocessor)
        recon_tgt = self._apply_post_processor(recon_tgt, self.features_and_labels_definition.reconstruction_targets_postprocessor)

        # drop nans
        features = pd_dropna(features)
        recon_tgt = pd_dropna(recon_tgt)

        # calculate the common index over all frames
        common_index = intersection_of_index(*features, recon_tgt)

        return FeaturesWithReconstructionTargets(
            [f.loc[common_index] for f in features],
            loc_if_not_none(recon_tgt, common_index),
            max([len(f) for f in features]) - len(common_index) + 1
        )

    def extract_labels(self) -> LabelsWithSampleWeights:
        label_types = self.features_and_labels_definition.label_type
        labels = self._extract_eventaually_nested_selectors(self.df, self.features_and_labels_definition.labels)
        weights = self._extract_eventaually_nested_selectors(self.df, self.features_and_labels_definition.sample_weights)
        gross_loss = call_if_not_none(get_pandas_object(self.df, self.features_and_labels_definition.gross_loss, type_map=self.type_map, **self.kwargs), 'dropna')

        # execute post-processors on labels and weights !!!
        labels = self._apply_post_processor(labels, self.features_and_labels_definition.labels_postprocessor)
        weights = self._apply_post_processor(weights, self.features_and_labels_definition.sample_weights_postprocessor)
        gross_loss = self._apply_post_processor(gross_loss, self.features_and_labels_definition.gross_loss_postprocessor)

        # drop nans
        labels = pd_dropna(labels)
        weights = pd_dropna(weights)
        gross_loss = pd_dropna(gross_loss)

        # calculate the common index over all frames
        common_index = intersection_of_index(*labels, *none_as_empty_list(weights), *none_as_empty_list(gross_loss))

        # set label types could be single element or list with same length as labels
        if label_types is not None:
            for i, lt in enumerate(make_same_length(label_types, labels)):
                if lt is not None:
                    labels[i] = labels[i].astype(lt)

        return LabelsWithSampleWeights(
            [l.loc[common_index] for l in labels],
            [loc_if_not_none(w, common_index) for w in weights] if weights is not None else None,
            loc_if_not_none(gross_loss, common_index),
        )

    # def extract_frames_for_fit(self, type_mapping: Dict[Type, callable] = {}, fitting_parameter: FittingParameter = FittingParameter(), verbose: int = 0, **kwargs: Dict) -> FeaturesWithLabels:
    #     frames = features_and_labels(
    #         df, extract_feature_labels_weights, type_map=type_mapping, fitting_parameter=fitting_parameter,
    #         verbose=verbose, **kwargs
    #     )
    #
    #     return frames
    #
    #
    # def extract_frames_for_backtest(self, type_mapping: Dict[Type, callable] = {}, tail: int = None, **kwargs: Dict) -> FeaturesWithLabels:
    #     min_required_samples = features_and_labels.min_required_samples
    #
    #     if tail is not None:
    #         if min_required_samples is not None:
    #             # just use the tail for feature engineering
    #             df = df[-(abs(tail) + (min_required_samples - 1)):]
    #         else:
    #             _log.warning("could not determine the minimum required data from the model")
    #
    #     frames = features_and_labels(
    #         df, extract_feature_labels_weights, type_map=type_mapping, **kwargs
    #     )
    #
    #     return frames
    #
    #
    # def extract_frames_for_predict(self, type_mapping: Dict[Type, callable] = {}, tail: int = None, **kwargs: Dict) -> FeaturesWithTargets:
    #     min_required_samples = features_and_labels.min_required_samples
    #
    #     if tail is not None:
    #         if min_required_samples is not None:
    #             # just use the tail for feature engineering
    #             df = df[-(abs(tail) + (min_required_samples - 1)):]
    #         else:
    #             _log.warning("could not determine the minimum required data from the model")
    #
    #     frames = features_and_labels(
    #         df, extract_features, type_map=type_mapping, **kwargs
    #     )
    #
    #     return frames

    def _extract_eventaually_nested_selectors(self, df, requested_selectors):
        if requested_selectors is None:
            return None

        # check direct access of a single item
        if is_in_index(requested_selectors, df.columns):
            requested_selectors = [requested_selectors]

        # check if requested_features are a nested data structure
        if all([is_in_index(rf, df.columns) for rf in requested_selectors if isinstance(rf, (List, Tuple))]):
            # it is NOT a nested data structure
            requested_selectors = [requested_selectors]

        # extract 2D features
        frames = [get_pandas_object(df, rf, type_map=self.type_map, **self.kwargs).dropna() for rf in requested_selectors]
        frames = [f.to_frame() if f.ndim <= 1 else f for f in frames]

        return frames

    def _apply_post_processor(self, frames, post_processors):
        if post_processors is None or (isinstance(post_processors, Iterable) and len(post_processors) <= 0):
            return frames

        return [
            call_callable_dynamic_args(pp, df=frames[i], **self.kwargs) if pp is not None else frames[i]
            for i, pp in enumerate(make_same_length(post_processors, frames))
        ]
