import abc
import dataclasses
import enum
import typing


class AbstractSelectionInfo(abc.ABC):
    pass


@dataclasses.dataclass
class SelectionInfoEmpty(AbstractSelectionInfo):
    pass


@dataclasses.dataclass
class SelectionInfoLabelMonoDirectionalRandom(AbstractSelectionInfo):
    from_label: typing.Optional[int] = dataclasses.field(default=None)


@dataclasses.dataclass
class SelectionInfoLabelMonoDirectional(AbstractSelectionInfo):
    from_label: typing.Optional[int] = dataclasses.field(default=None)
    to_label: typing.Optional[int] = dataclasses.field(default=None)


@dataclasses.dataclass
class SelectionInfoLabelBiDirectionalMirrored(AbstractSelectionInfo):
    label_a: typing.Optional[int] = dataclasses.field(default=None)
    label_b: typing.Optional[int] = dataclasses.field(default=None)


def get_value_from_percentage(perc: float, value: int) -> int:
    return round(value * perc / 100)


@dataclasses.dataclass
class AbstractPerformInfo(abc.ABC):
    perc_data_points: float = dataclasses.field(default=0.0)

    def get_number_of_data_points(self, total_number: int) -> int:
        return get_value_from_percentage(perc=self.perc_data_points, value=total_number)

    @abc.abstractmethod
    def get_info_as_dict(self) -> typing.Dict[str, typing.Any]:
        pass

    @abc.abstractmethod
    def get_info_clean_as_dict(self) -> typing.Dict[str, typing.Any]:
        pass


PERC_POINTS = 'Perc_Points'
PERC_FEATURES = 'Perc_Features'

# todo in dataset generator will just add perc_points and perc_features directly, removing
# these abstract method.


@dataclasses.dataclass
class PerformInfoLabelOnly(AbstractPerformInfo):

    def get_info_as_dict(self) -> typing.Dict[str, typing.Any]:
        return {PERC_POINTS: self.perc_data_points, PERC_FEATURES: 0.0}

    def get_info_clean_as_dict(self) -> typing.Dict[str, typing.Any]:
        return {PERC_POINTS: 0.0, PERC_FEATURES: 0.0}


@dataclasses.dataclass
class PerformInfoMonoDirectional(SelectionInfoLabelMonoDirectional, PerformInfoLabelOnly):
    pass


class PerformInfoEmpty(PerformInfoLabelOnly):
    pass


@dataclasses.dataclass
class PerformInfoBiDirectionalMirrored(SelectionInfoLabelBiDirectionalMirrored, PerformInfoLabelOnly):
    pass


@typing.runtime_checkable
class PerformInfoProtocol(typing.Protocol):

    def get_info_as_dict(self) -> typing.Dict[str, typing.Any]:
        pass

    def get_info_clean_as_dict(self) -> typing.Dict[str, typing.Any]:
        pass


class ModifiedPartOfPoints(enum.Enum):
    X = 'X'
    y = 'y'
    X_y = 'X_Y'


PoisoningInfo_D = typing.Dict[str, typing.Union[str, int, float, bool]]