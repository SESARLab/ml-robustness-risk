import dataclasses
import typing

import dacite


from . import base, performers, selectors, wrapper


TSelector = typing.TypeVar('TSelector', bound=selectors.AbstractSelector)
TPerformer = typing.TypeVar('TPerformer', bound=performers.AbstractPerformer)


@dataclasses.dataclass
class PoisoningGenerationInput:
    """
    Attributes

    selector: TSelector
    performer: TPerformer

    perform_info_clazz: typing.Type[base.AbstractPerformInfo] info on the perturbation algorithm
    perform_info_kwargs: typing.Dict[str, any] info on the perturbation algorithm

    selection_info_clazz: typing.Type[base.AbstractSelectionInfo]
    selection_info_kwargs: typing.Dict[str, any]

    perc_data_points: typing.Sequence[float]
    perc_features: typing.Sequence[float] = dataclasses.field(default_factory=list)
    columns: typing.Optional[typing.List[str]] = dataclasses.field(default=None)
    """
    selector: TSelector
    performer: TPerformer
    # this contains the information to pass to the algorithm
    perform_info_clazz: typing.Type[base.AbstractPerformInfo]
    perform_info_kwargs: typing.Dict[str, any]

    selection_info_clazz: typing.Type[base.AbstractSelectionInfo]
    selection_info_kwargs: typing.Dict[str, any]

    perc_data_points: typing.Sequence[float]
    # so far it is mostly ignored
    perc_features: typing.Sequence[float] = dataclasses.field(default_factory=list)
    columns: typing.Optional[typing.List[str]] = dataclasses.field(default=None)

    def __post_init__(self):
        self.perc_features = [0.0 for _ in range(len(self.perc_data_points))]
        # do some sanity checks on the content of two kwargs. To avoid strange things,
        # we do raise an Exception.
        k_selection_s = set(self.selection_info_kwargs.keys())
        k_perform_s = set( self.perform_info_kwargs.keys())
        for common_key in k_selection_s.intersection(k_perform_s):
            if self.selection_info_kwargs[common_key] != self.perform_info_kwargs[common_key]:
                raise ValueError(f'Inconsistency in the configuration of poisoning algorithm for key {common_key}. '
                                 f'Selection got: {self.selection_info_kwargs[common_key]}, '
                                 f'perform got {self.perform_info_kwargs[common_key]}')

    def generate_from_sequence(self) -> typing.Tuple[typing.List[typing.Tuple[float, float]], typing.List[wrapper.Poisoning[TSelector, TPerformer]]]:

        wrappers = []
        poisoning_points = []

        for single_perc_points, single_perc_features in zip(self.perc_data_points, self.perc_features):
            poisoning_points.append((float(single_perc_points), float(single_perc_features)))

            config_raw = {
                'perc_data_points': single_perc_points,
                **self.perform_info_kwargs
            }

            # if the info class supports features poisoning as well, then we add such values as well.
            if hasattr(self.perform_info_clazz, 'perc_features'):
                config_raw['perc_features'] = single_perc_features

            perform_info_parsed = dacite.from_dict(data_class=self.perform_info_clazz, data=config_raw)
            selection_info_parsed = dacite.from_dict(data_class=self.selection_info_clazz,
                                                     data=self.selection_info_kwargs)

            w = wrapper.Poisoning(perform_info=perform_info_parsed, selection_info=selection_info_parsed,
                                  selector=self.selector, performer=self.performer)
            wrappers.append(w)

        return poisoning_points, wrappers
