import abc
import dataclasses
import os

import mashumaro

import experiments
from . import (base, raw_dataset, raw_exp_monolithic_models, raw_exp_ensemble_plain_advanced,
               raw_exp_ensemble_risk, raw_exp_iop)


class AbstractTopLevelAction(abc.ABC):
    """
    Just a blank interface all top-level actions must implement in order to be called from the outside.
    """

    @abc.abstractmethod
    def do(self):
        pass


class AbstractTopLevelExperiment(abc.ABC):
    """
    The abstract method(s) it contains are just to avoid
    the type checker complaining about those properties missing here (since are attributes present
    in the child class(es).
    """

    @property
    @abc.abstractmethod
    def parse(self):
        pass

    def _parse_and_exec(self, expand_results: bool):
        """
        :param expand_results: whether the result of the experiment should
        be passed as is (e.g., in the case of IoP because it is a single returned value),
        or use python expression expansion (e.g., exp ensemble plain and risk).
        :return:
        """

        export_config = self.export_config.parse()
        export_config.base_directory = os.path.join(self.base_output_directory, base.BASE_OUTPUT_DIR_OUTPUT)
        # if it already exists and no override permitted, let's stop here.
        if os.path.exists(export_config.base_directory):
            # safety check just because
            if hasattr(export_config, 'exists_ok'):
                if not export_config.exists_ok:
                    raise ValueError(f'Output directory {export_config.base_directory} '
                                     f'exists and results override is forbidden. So we exit.')

        parsed: experiments.AbstractExperiment = self.parse()
        results = parsed.do()

        if expand_results:
            analyzed_results = parsed.analysis_class.from_results(*results)
        else:
            analyzed_results = parsed.analysis_class.from_results(results)
        analyzed_results.export(export_config)
        return analyzed_results


@dataclasses.dataclass
class TopLevelExpEnsembleRisk(raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw,
                              AbstractTopLevelExperiment,
                              AbstractTopLevelAction,
                              mashumaro.DataClassDictMixin):

    def do(self):
        self._parse_and_exec(expand_results=True)


@dataclasses.dataclass
class TopLevelExpMonolithicModels(raw_exp_monolithic_models.ExperimentMonolithicModelRaw,
                                  AbstractTopLevelExperiment,
                                  AbstractTopLevelAction,
                                  mashumaro.DataClassDictMixin):
    def do(self):
        self._parse_and_exec(expand_results=True)


@dataclasses.dataclass
class TopLevelExpEnsemblePlainAdvanced(raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw,
                                       AbstractTopLevelExperiment,
                                       AbstractTopLevelAction,
                                       mashumaro.DataClassDictMixin):

    def do(self):
        self._parse_and_exec(expand_results=True)


@dataclasses.dataclass
class TopLevelExpIop(raw_exp_iop.ExperimentIoPRaw,
                     AbstractTopLevelExperiment,
                     AbstractTopLevelAction,
                     mashumaro.DataClassDictMixin):

    def do(self):
        self._parse_and_exec(expand_results=False)


@dataclasses.dataclass
class TopLevelDataset(raw_dataset.DatasetToPoisonRaw,
                      AbstractTopLevelAction,
                      mashumaro.DataClassDictMixin):
    base_output_directory: str = dataclasses.field(default='')
    exists_ok: bool = dataclasses.field(default=False)

    def do(self):
        self.parse_and_load(base_output_directory=self.base_output_directory,
                            exists_ok=self.exists_ok)
