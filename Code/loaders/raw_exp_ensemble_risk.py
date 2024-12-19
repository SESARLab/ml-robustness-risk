import dataclasses
import typing

import mashumaro

import experiments
from . import base, raw_exp_abstract, raw_pipe


@dataclasses.dataclass
class ExportConfigExpEnsembleRiskRaw(mashumaro.DataClassDictMixin,
                                     base.RawToParsed[experiments.ExportConfigExpEnsembleRisk]):
    exists_ok: bool = dataclasses.field(default=False)

    def parse(self) -> experiments.ExportConfigExpEnsembleRisk:
        # also here, we'll add the base directory later.
        return experiments.ExportConfigExpEnsembleRisk(exists_ok=self.exists_ok)


@dataclasses.dataclass
class ExperimentEnsembleRiskRaw(mashumaro.DataClassDictMixin,
                                base.RawToParsed[experiments.ExperimentEnsembleRisk],
                                raw_exp_abstract.AbstractExperimentWithPipelines):
    know_all_pipelines: typing.List[raw_pipe.PipelineRaw] = dataclasses.field(default_factory=list)

    export_config: ExportConfigExpEnsembleRiskRaw = dataclasses.field(default_factory=ExportConfigExpEnsembleRiskRaw)
    monolithic_model: base.FuncPair = dataclasses.field(default_factory=base.FuncPair)

    def parse(self) -> experiments.ExperimentEnsembleRisk:
        dg, pipelines = self.get_dg_and_pipelines()
        baselines = self.parse_pipelines(pipelines=self.know_all_pipelines)

        return experiments.ExperimentEnsembleRisk.from_dataset_generator(dg=dg, pipelines=pipelines,
                                                                         monolithic_model=self.monolithic_model.parse(),
                                                                         repetitions=self.repetitions,
                                                                         ground_truth_pipelines=baselines)
