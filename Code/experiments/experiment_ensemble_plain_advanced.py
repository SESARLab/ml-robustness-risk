import dataclasses
import typing

import numpy as np
import xarray as xr

from . import base, dataset_generator, experiment_common as common
import pipe
import utils


@dataclasses.dataclass
class ExportConfigExpEnsemblePlainAdvanced(base.AbstractExportConfig):
    pass


# this is necessary
@dataclasses.dataclass
class AnalyzedResultsEnsemblePlainAdvanced(common.AnalyzedResultsEnsembleCommon):
    pass


TEstimator = typing.TypeVar('TEstimator', bound=utils.EstimatorProtocol)


class ExperimentEnsemblePlainAdvanced(common.AbstractCommonExperiment):

    def __init__(self,
                 repetitions: int,
                 monolithic_model: TEstimator,
                 X_train_clean: np.ndarray,
                 y_train_clean: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 poisoned_datasets: xr.Dataset,
                 pipelines: typing.List[pipe.ExtPipeline],
                 columns: typing.Optional[typing.List[str]] = None
                 ):
        super().__init__(repetitions=repetitions, monolithic_model=monolithic_model,
                         X_train_clean=X_train_clean, y_train_clean=y_train_clean,
                         X_test=X_test, y_test=y_test, poisoned_datasets=poisoned_datasets,
                         pipelines=pipelines, columns=columns)

    @property
    def analysis_class(self) -> typing.Type[AnalyzedResultsEnsemblePlainAdvanced]:
        return AnalyzedResultsEnsemblePlainAdvanced

    @staticmethod
    def from_dataset_generator(dg: dataset_generator.DatasetGenerator,
                               pipelines: typing.List[pipe.ExtPipeline],
                               monolithic_model: TEstimator,
                               repetitions: int) -> "ExperimentEnsemblePlainAdvanced":
        return ExperimentEnsemblePlainAdvanced(X_test=dg.X_test, y_test=dg.y_test, X_train_clean=dg.X_train_clean,
                                               y_train_clean=dg.y_train_clean, monolithic_model=monolithic_model,
                                               pipelines=pipelines, poisoned_datasets=dg.all_datasets,
                                               columns=dg.columns, repetitions=repetitions)
