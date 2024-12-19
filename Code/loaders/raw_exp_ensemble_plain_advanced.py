import dataclasses

import mashumaro

from . import base, raw_exp_abstract
import experiments


@dataclasses.dataclass
class ExportConfigExpEnsemblePlainAdvancedRaw(mashumaro.DataClassDictMixin,
                                              base.RawToParsed[experiments.ExportConfigExpEnsemblePlainAdvanced]):
    exists_ok: bool = dataclasses.field(default=False)

    def parse(self) -> experiments.ExportConfigExpEnsemblePlainAdvanced:
        # also here, we'll add the base directory later.
        return experiments.ExportConfigExpEnsemblePlainAdvanced(exists_ok=self.exists_ok)


@dataclasses.dataclass
class ExperimentEnsemblePlainAdvancedRaw(mashumaro.DataClassDictMixin,
                                         base.RawToParsed[experiments.ExperimentEnsemblePlainAdvanced],
                                         raw_exp_abstract.AbstractExperimentWithPipelines):
    export_config: ExportConfigExpEnsemblePlainAdvancedRaw = dataclasses.field(
        default_factory=ExportConfigExpEnsemblePlainAdvancedRaw)
    monolithic_model: base.FuncPair = dataclasses.field(default_factory=base.FuncPair)

    def parse(self) -> experiments.ExperimentEnsemblePlainAdvanced:
        dg, pipelines = self.get_dg_and_pipelines()
        # base_model = base.load_func(self.base_model_name, self.base_model_kwargs)
        return experiments.ExperimentEnsemblePlainAdvanced.from_dataset_generator(dg=dg, pipelines=pipelines,
                                                                                  monolithic_model=self.monolithic_model.parse(),
                                                                                  repetitions=self.repetitions)
