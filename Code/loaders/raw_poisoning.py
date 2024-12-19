import dataclasses
import typing

import mashumaro

from . import base
import poisoning


@dataclasses.dataclass
class SelectorRaw(mashumaro.DataClassDictMixin, base.RawToParsed[poisoning.AbstractSelector]):

    name: str
    init_kwargs: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def parse(self) -> poisoning.AbstractSelector:
        # first we load the class and its kwargs.
        obj: poisoning.AbstractSelector = base.load_func(self.name, self.init_kwargs)
        return obj


@dataclasses.dataclass
class PerformerRaw(mashumaro.DataClassDictMixin, base.RawToParsed[poisoning.AbstractPerformer]):
    # the class name
    name: str
    init_kwargs: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    # it will be converted in the appropriate type when we convert this object first,
    # because we can't know in advance to which type this needs to be converted (it depends on "name").
    # performing_info: typing.Optional[typing.Union[poisoning.PoisoningInfoMonoDirectional, poisoning.PoisoningInfoBiDirectionalMirrored]] = dataclasses.field(default=None)

    def parse(self) -> poisoning.AbstractPerformer:
        # first we load the class and its kwargs.
        return base.load_func(self.name, self.init_kwargs)


@dataclasses.dataclass
class PoisoningGenerationInfoRaw(mashumaro.DataClassDictMixin, base.RawToParsed[poisoning.PoisoningGenerationInput]):
    selector: SelectorRaw
    performer: PerformerRaw
    perform_info_clazz: str

    selection_info_clazz: str

    perc_data_points: typing.Sequence[float]
    perc_features: typing.Sequence[float] = dataclasses.field(default_factory=list)
    columns: typing.Optional[typing.List[str]] = dataclasses.field(default=None)
    # shuffle: bool = dataclasses.field(default=True)
    # train_split: float = dataclasses.field(default=.75)

    perform_info_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = dataclasses.field(default_factory=dict)
    selection_info_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = dataclasses.field(default_factory=dict)

    def parse(self) -> poisoning.PoisoningGenerationInput:
        selector = self.selector.parse()
        performer = self.performer.parse()
        # load the class holding the poisoning information. Note that this is the CLASS only,
        # i.e., the name of the class rather than an object of this type.
        perform_info_clazz = base.load_func(self.perform_info_clazz, None)
        perform_info_kwargs = base.fill_kwargs(self.perform_info_kwargs)

        selection_info_clazz = base.load_func(self.selection_info_clazz, None)
        selection_info_kwargs = base.fill_kwargs(self.selection_info_kwargs)

        return poisoning.PoisoningGenerationInput(
            columns=self.columns, selector=selector, performer=performer,
            perform_info_clazz=perform_info_clazz, perform_info_kwargs=perform_info_kwargs,
            selection_info_clazz=selection_info_clazz, selection_info_kwargs=selection_info_kwargs,
            perc_features=self.perc_features, perc_data_points=self.perc_data_points)
