import dataclasses
import typing

import mashumaro

import experiments
from . import base, raw_exp_abstract


@dataclasses.dataclass
class ExportConfigExpBaseModelsRaw(mashumaro.DataClassDictMixin,
                                   base.RawToParsed[experiments.ExportConfigBaseModels]):
    exists_ok: bool = dataclasses.field(default=False)

    def parse(self) -> experiments.ExportConfigBaseModels:
        # also here, we'll add the base directory later.
        return experiments.ExportConfigBaseModels(exists_ok=self.exists_ok)


@dataclasses.dataclass
class ExperimentMonolithicModelRaw(mashumaro.DataClassDictMixin,
                                   base.RawToParsed[experiments.ExperimentMonolithicModels],
                                   raw_exp_abstract.AbstractExperiment):

    monolithic_models: typing.Dict[str, base.FuncPair] = dataclasses.field(default_factory=list)
    export_config: ExportConfigExpBaseModelsRaw = dataclasses.field(default_factory=ExportConfigExpBaseModelsRaw)

    def parse(self) -> experiments.ExperimentMonolithicModels:
        dg = self.get_dg()
        return experiments.ExperimentMonolithicModels.from_dataset_generator(
            dg=dg, monolithic_models=[(k, v.parse()) for (k, v) in self.monolithic_models.items()],
            repetitions=self.repetitions)
