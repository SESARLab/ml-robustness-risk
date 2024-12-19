import dataclasses

import mashumaro

from . import base, raw_exp_abstract
import experiments


@dataclasses.dataclass
class ExportConfigExpIoPRaw(mashumaro.DataClassDictMixin,
                            base.RawToParsed[experiments.ExportConfigIoP]):
    exists_ok: bool = dataclasses.field(default=False)
    export_also_raw_results: bool = dataclasses.field(default=False)
    export_also_figures: bool = dataclasses.field(default=False)
    export_png: bool = dataclasses.field(default=False)
    export_html: bool = dataclasses.field(default=False)

    def parse(self) -> experiments.ExportConfigIoP:
        # we'll fill the base directory later.
        return experiments.ExportConfigIoP(exists_ok=self.exists_ok,
                                           export_also_iops=self.export_also_raw_results,
                                           export_also_figures=self.export_also_figures,
                                           export_png=self.export_png, export_html=self.export_html)


@dataclasses.dataclass
class ExperimentIoPRaw(mashumaro.DataClassDictMixin,
                       base.RawToParsed[experiments.ExperimentIoP],
                       raw_exp_abstract.AbstractExperimentWithPipelines):

    export_config: ExportConfigExpIoPRaw = dataclasses.field(default_factory=ExportConfigExpIoPRaw)

    def parse(self) -> experiments.ExperimentIoP:
        dg, pipelines = self.get_dg_and_pipelines()
        # now check that we have at least the step to evaluate in any pipelines.
        for p in pipelines:
            if p.steps_to_evaluate is None or len(p.steps_to_evaluate) == 0:
                raise ValueError(f'Pipeline {p.name} does not specify \'steps_to_evaluate\'')
        return experiments.ExperimentIoP.from_dataset_generator(
            dg=dg, pipelines=pipelines, repetitions=self.repetitions,
            keep_also_figures=self.export_config.export_also_figures,
            keep_also_iops=self.export_config.export_also_raw_results)
