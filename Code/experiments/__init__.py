from dask.distributed import Client, LocalCluster

# # replace with whichever cluster class you're using
# # https://docs.dask.org/en/stable/deploying.html#distributed-computing
# cluster = LocalCluster()
# # connect client to your cluster
# client = Client(cluster)


from .base import DIR_DATASET_NAME_EXPORT_CSV, DIR_DATASET_NAME_EXPORT_BINARY, \
    FILE_NAME_DATASET_PREFIX_CLEAN, FILE_NAME_DATASET_PREFIX_TEST, FILE_NAME_DATASET_PREFIX_POISONED, \
    AbstractExperiment,  \
    FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, \
    FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF, FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED

from . dataset_generator import DatasetGenerator
from . experiment_monolithic_models import ExperimentMonolithicModels, ExportConfigBaseModels, AnalyzedResultMonolithicModels
from . experiment_iop import ExperimentIoP, ExportConfigIoP, AnalyzedResultsIoP
# from . experiment_ensemble_plain import (ExperimentEnsemblePlain, ExportConfigExpEnsemblePlain,
#                                          AnalyzedResultsEnsemblePlain)
from . experiment_ensemble_risk import ExperimentEnsembleRisk, ExportConfigExpEnsembleRisk, AnalyzedResultsEnsembleRisk
from . experiment_ensemble_plain_advanced import ExperimentEnsemblePlainAdvanced, ExportConfigExpEnsemblePlainAdvanced, \
    AnalyzedResultsEnsemblePlainAdvanced

from .experiment_common import TestSetType

___all__ = [
    AbstractExperiment,
    DIR_DATASET_NAME_EXPORT_CSV, DIR_DATASET_NAME_EXPORT_BINARY, FILE_NAME_DATASET_PREFIX_CLEAN,
    FILE_NAME_DATASET_PREFIX_TEST, FILE_NAME_DATASET_PREFIX_POISONED,
    FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF,
    DatasetGenerator,
    ExperimentIoP, ExportConfigIoP, AnalyzedResultsIoP,
    ExperimentMonolithicModels, ExportConfigBaseModels, AnalyzedResultMonolithicModels,
    # ExperimentEnsemblePlain, ExportConfigExpEnsemblePlain, AnalyzedResultsEnsemblePlain,
    ExperimentEnsembleRisk, ExportConfigExpEnsembleRisk, ExportConfigExpEnsembleRisk, AnalyzedResultsEnsembleRisk,
    FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED,
    TestSetType
]
