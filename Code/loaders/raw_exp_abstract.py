import abc
import dataclasses
import os
import typing

import utils
from . import base, raw_dataset, raw_pipe
import experiments
import pipe


@dataclasses.dataclass
class AbstractExperiment(abc.ABC):
    repetitions: int
    base_output_directory: str
    dataset_exists_ok: bool
    dataset_config_poisoned: typing.Optional[raw_dataset.DatasetAlreadyPoisonedRaw] = dataclasses.field(default=None)
    dataset_config_to_poison: typing.Optional[raw_dataset.DatasetToPoisonRaw] = dataclasses.field(default=None)

    def __post_init__(self):
        if (self.dataset_config_poisoned is None and self.dataset_config_to_poison is None) or \
                (self.dataset_config_poisoned is not None and self.dataset_config_to_poison is not None):
            raise ValueError('Only one between dataset_config_to_poison and dataset_config_poisoned can be not None')

    @property
    def dataset_config(self) -> typing.Union[raw_dataset.DatasetToPoisonRaw, raw_dataset.DatasetAlreadyPoisonedRaw]:
        if self.dataset_config_poisoned is not None:
            return self.dataset_config_poisoned
        return self.dataset_config_to_poison

    def get_dg(self):
        return self.dataset_config.parse_and_load(
            exists_ok=self.dataset_exists_ok,
            base_output_directory=os.path.join(self.base_output_directory, base.BASE_OUTPUT_DIR_DATASET))


@dataclasses.dataclass
class AbstractExperimentWithPipelines(AbstractExperiment, abc.ABC):
    pipelines: typing.List[raw_pipe.PipelineRaw] = dataclasses.field(default_factory=list)

    @staticmethod
    def parse_pipelines(pipelines: typing.List[raw_pipe.PipelineRaw]) -> typing.List[pipe.ExtPipeline]:
        pipelines = [p.parse() for p in pipelines]
        # check that pipeline names are unique
        # set_names = {p.name for p in pipelines}

        duplicates = utils.get_duplicates([p.name for p in pipelines])
        if len(duplicates) > 0:
            raise ValueError(f'There are pipeline with duplicated names.\n'
                             f'duplicates names are {duplicates}')

        return pipelines

    def get_dg_and_pipelines(self) -> typing.Tuple[experiments.DatasetGenerator, typing.List[pipe.ExtPipeline]]:
        pipelines = self.parse_pipelines(pipelines=self.pipelines)
        dg = self.get_dg()
        return dg, pipelines
