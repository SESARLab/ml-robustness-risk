import abc
import dataclasses
import functools
import typing

import numpy as np
import pandas as pd
from sklearn import metrics
import xarray as xr

import const
import pipe
import utils

# METRIC_NAME_ACCURACY = 'Accuracy'
# METRIC_NAME_F1 = 'F1'
# METRIC_NAME_PRECISION = 'Precision'
# METRIC_NAME_RECALL = 'Recall'

METRICS = [(const.METRIC_NAME_ACCURACY, metrics.accuracy_score),
           (const.METRIC_NAME_PRECISION, metrics.precision_score),
           (const.METRIC_NAME_RECALL, metrics.recall_score),
           #    (METRIC_NAME_F1, metrics.f1_score)
           ]

KEY_TRAIN_RESULT_PERC_DATA_POINTS = 'Perc_Point'
KEY_TRAIN_RESULT_PERC_FEATURES = 'Perc_Feature'
KEY_TRAIN_RESULT_X = 'X'
KEY_TRAIN_RESULT_Y = 'y'

METRICS_NAME = [m[0] for m in METRICS]
METRICS_FUNC = [m[1] for m in METRICS]

# KEY_TRAIN_RESULT_DELTA_ACCURACY = 'Delta(Acc)'
# KEY_TRAIN_RESULT_DELTA_F1 = 'Delta(F1)'
# KEY_TRAIN_RESULT_DELTA_PRECISION = 'Delta(Prec)'
# KEY_TRAIN_RESULT_DELTA_RECALL = 'Delta(Rec)'

# KEY_TRAIN_RESULT_MONO_ACCURACY = 'Mono_Accuracy'
# KEY_TRAIN_RESULT_MONO_F1 = 'Mono_F1'
# KEY_TRAIN_RESULT_MONO_PRECISION = 'Mono_Precision'
# KEY_TRAIN_RESULT_MONO_RECALL = 'Mono_Recall'
#
# KEY_TRAIN_RESULT_ENSEMBLE_ACCURACY = 'Ensemble_Accuracy'
# KEY_TRAIN_RESULT_ENSEMBLE_F1 = 'Ensemble_F1'
# KEY_TRAIN_RESULT_ENSEMBLE_PRECISION = 'Ensemble_Precision'
# KEY_TRAIN_RESULT_ENSEMBLE_RECALL = 'Ensemble_Recall'

# KEY_TRAIN_RESULT_ENSEMBLE_POISONED_ACCURACY = 'Ensemble_Poisoned_Accuracy'
# KEY_TRAIN_RESULT_ENSEMBLE_POISONED_F1 = 'Ensemble_Poisoned_F1'
# KEY_TRAIN_RESULT_ENSEMBLE_POISONED_PRECISION = 'Ensemble_Poisoned_Precision'
# KEY_TRAIN_RESULT_ENSEMBLE_POISONED_RECALL = 'Ensemble_Poisoned_Recall'
#
# KEY_TRAIN_RESULT_MONO_POISONED_ACCURACY = 'Mono_Poisoned_Accuracy'
# KEY_TRAIN_RESULT_MONO_POISONED_F1 = 'Mono_Poisoned_F1'
# KEY_TRAIN_RESULT_MONO_POISONED_PRECISION = 'Mono_Poisoned_Precision'
# KEY_TRAIN_RESULT_MONO_POISONED_RECALL = 'Mono_Poisoned_Recall'

# KEY_TRAIN_RESULT_MAJORITY_COUNT_AVG = 'Majority_Avg'
# KEY_TRAIN_RESULT_MAJORITY_COUNT_STD = 'Majority_Std'
# KEY_TRAIN_RESULT_MAJORITY_COUNT_N_DISCORDANT = 'Majority_Count_Disc'

# KEY_TRAIN_ASSIGNMENT_COLUMN_ASSIGNMENT = 'Assignment'
# KEY_TRAIN_ASSIGNMENT_COLUMN_POISONED = 'Poisoned'

# EXPORTED_DATASET_POSTFIX_POISONED = 'poisoned'
# EXPORTED_DATASET_POSTFIX_CLEAN_TRAIN = 'train_clean'
# EXPORTED_DATASET_POSTFIX_CLEAN_TEST = 'test_clean'


# def delta_only(df: pd.DataFrame) -> pd.DataFrame:
#     columns = [c for c in df.columns if 'Delta' in c]
#     return df[columns]




# COORD_AVG = 'AVG'
# COORD_STD = 'STD'

# COORD_MAJORITY_COUNT_AVG = 'AVG(MAJORITY)'
# COORD_MAJORITY_COUNT_STD = 'STD(MAJORITY)'
# COORD_MAJORITY_COUNT_DISCORDANT = 'COUNT(MAJORITY_DISC)'

# # given an assignment, it models the largest
# KEY_ASSIGNMENT_RISK_MAX_DIFFERENCES = 'Assignment_Max_Differences'


# def avg_and_std_from_seq(results: typing.Sequence[xr.DataArray]) -> typing.Tuple[xr.DataArray, xr.DataArray]:
#
#     concat = xr.concat(results, dim='y')
#
#     # these operations are a bit tricky since we are working with 1d arrays,
#     # the point is that
#     # a 1d array has dimension 'x'
#     # when we concat two 1d arrays on dimension y, the dimension 'x' corresponds to the "columns"
#     # while the 'x' to the rows.
#     # This is actually correct but still an issue for our purposes.
#     # So we rename.
#
#     concat = concat.rename({'x': 'y', 'y': 'x'})
#
#     # here we rename because the operation performed on a dimension means "reduce" on such a dimension.
#     # As a consequence, only the other dimension remains. Again, this is correct,
#     # but since we are ending up with two 1d arrays with dimension named 'y' this is an issue.
#     avg = concat.mean(dim='x', keep_attrs=True).rename({'y': 'x'})
#     std = concat.std(dim='x', keep_attrs=True).rename({'y': 'x'})
#
#     agg = xr.concat([avg, std], dim='y').rename({'x': 'y', 'y': 'x'})
#
#     agg = agg.assign_coords({
#             'x': [COORD_AVG, COORD_STD]
#         })
#
#     return concat, agg



# COORD_NAME_ASSIGNMENT: str = 'Assignment'

EXPORT_NAME_PREFIX_DATASET_CLEAN = 'Clean'
EXPORT_NAME_PREFIX_DATASET_ALL = 'All'
EXPORT_NAME_PREFIX_DATASET_POISONED = 'Poisoned'

KEY_DATASET_CLEAN = 'Clean'
KEY_DATASET_POISONED = 'Poisoned'

EXP_DIR_TEST_SET_CLEAN = 'TestSetClean'
EXP_DIR_TRAINING_SET_CLEAN = 'TrainingSetClean'

EXPORT_NAME_PREFIX_TEST_SET_TYPE_TEST_SET_CLEAN = 'test_set_clean'
EXPORT_NAME_PREFIX_TEST_SET_TYPE_TRAINING_SET_CLEAN = 'training_set_clean'

DIR_NAME_EXPORT_AGGREGATED = 'Aggregated'
DIR_NAME_EXPORT_NOT_AGGREGATED = 'NonAggregated'
DIR_NAME_EXPORT_IOP_IMPORTANT = 'IoP_Important'
DIR_NAME_EXPORT_IOP_ALL = 'IoP_All'

# FILE_NAME_EXPORT_DELTA_BASE = 'delta_base'
FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF = 'mono_vanilla_delta_self'
FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF = 'mono_oracled_delta_self'
# FILE_NAME_EXPORT_DELTA_ENSEMBLE = 'delta_ensemble'
FILE_NAME_EXPORT_ENSEMBLE_ASSIGNMENT = 'assignment'
FILE_NAME_EXPORT_MONO_VANILLA_QUALITY = 'mono_vanilla_quality'
FILE_NAME_EXPORT_MONO_ORACLED_QUALITY = 'mono_oracled_quality'
FILE_NAME_EXPORT_ENSEMBLE_QUALITY = 'ensemble_quality'

FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_VANILLA = 'ensemble_delta_ref_mono_vanilla'
FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_ORACLED = 'ensemble_delta_ref_mono_oracle'
FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED = 'mono_vanilla_delta_ref_mono_oracle'
FILE_NAME_EXPORT_ENSEMBLE_DELTA_SELF = 'delta_self_ensemble'

DIR_NAME_EXPORT_NOT_AGGREGATED_DELTA_MONOLITHIC = 'Delta_Monolithic'
DIR_NAME_EXPORT_NOT_AGGREGATED_DELTA_ENSEMBLE = 'Delta_Ensemble'
DIR_NAME_EXPORT_NOT_AGGREGATED_MODEL_MONOLITHIC = 'Model_Monolithic'
DIR_NAME_EXPORT_NOT_AGGREGATED_MODEL_ENSEMBLE = 'Model_Ensemble'
DIR_NAME_EXPORT_NOT_AGGREGATED_ASSIGNMENT = 'Assignment'

FILE_NAME_DATASET_PREFIX_CLEAN = 'clean'
FILE_NAME_DATASET_PREFIX_TEST = 'test'
FILE_NAME_DATASET_PREFIX_POISONED = 'poisoned'

DIR_DATASET_NAME_EXPORT_CSV = 'CSV'
DIR_DATASET_NAME_EXPORT_BINARY = 'Binary'


class AnalyzedResultsProto(typing.Protocol):
    """
    Interface modeling the functionalities different classes
    containing the analyzed results produced by any experiments.
    """

    @staticmethod
    def from_results(*vals):
        pass

    def export(self, config: typing.Any):
        pass


class AbstractExperiment(abc.ABC):

    def __init__(self, repetitions: int,
                 poisoned_datasets: xr.Dataset,
                 clean_dataset_attrs: dict,
                 columns: typing.Optional[typing.List[str]] = None, ):
        self.repetitions = repetitions
        self.poisoned_datasets = poisoned_datasets

        got_column = self.poisoned_datasets[list(self.poisoned_datasets.keys())[0]].coords['y'].values
        got_column = [got for got in got_column if got not in const.DG_IRRELEVANT_COLUMNS]

        self.columns = utils.check_and_get_columns(expected=len(got_column), got=columns)
        self.clean_dataset_attrs = clean_dataset_attrs

    @property
    @abc.abstractmethod
    def analysis_class(self) -> typing.Type[AnalyzedResultsProto]:
        pass

    @abc.abstractmethod
    def do(self):
        pass


COORD_OUTPUT_IOP_COLS = 'Output'
KEY_ATTR_PIPE_NAME = 'Pipe'

KEY_MAJORITY_COUNT_AVG = 'Majority_Avg'
KEY_MAJORITY_COUNT_STD = 'Majority_Std'
KEY_COUNT_N_DISCORDANT = 'Majority_Count_Disc'

KEY_ASSIGNMENT = 'Assignment'
KEY_POISONED = 'Poisoned'



# MONOLITHIC_VANILLA_PIPELINE_NAME = 'MONO_VANILLA'
# MONOLITHIC_ORACLED_PIPELINE_NAME = 'MONO_ORACLED'

KEY_DELTA = 'Delta'

KEY_ASSIGNMENT_RECALL = f'{const.METRIC_NAME_RECALL}(Assignment)'
KEY_ASSIGNMENT_FULLNESS_AVG = f'{const.PREFIX_AVG}(Fullness)'
KEY_ASSIGNMENT_FULLNESS_STD = f'{const.PREFIX_STD}(Fullness)'
KEY_ASSIGNMENT_DIVERSITY_AVG = f'{const.PREFIX_AVG}(Diversity)'
# KEY_ASSIGNMENT_DIVERSITY_STD = f'{PREFIX_STD}(Diversity)'
KEY_RISK_QUALITY = 'Risk'
KEY_ASSIGNMENT_POISONING = 'PoisoningAssignment'

KEYS_ASSIGNMENT_BASIC = [KEY_ASSIGNMENT_POISONING, KEY_ASSIGNMENT_FULLNESS_AVG, KEY_ASSIGNMENT_FULLNESS_STD,
                         KEY_ASSIGNMENT_DIVERSITY_AVG,  # KEY_ASSIGNMENT_DIVERSITY_STD,
                         KEY_MAJORITY_COUNT_AVG, KEY_MAJORITY_COUNT_STD, KEY_COUNT_N_DISCORDANT]
KEYS_ASSIGNMENT_RISK = [KEY_ASSIGNMENT_POISONING, KEY_ASSIGNMENT_RECALL, KEY_ASSIGNMENT_FULLNESS_AVG,
                        KEY_ASSIGNMENT_DIVERSITY_AVG,  # KEY_ASSIGNMENT_DIVERSITY_STD,
                        KEY_ASSIGNMENT_FULLNESS_STD, KEY_MAJORITY_COUNT_AVG, KEY_MAJORITY_COUNT_STD,
                        KEY_COUNT_N_DISCORDANT]

# # EXP_DIR_MERGED = 'Merged'
EXP_DIR_MERGED = 'Merged'
# # EXP_DIR_MODEL_QUALITY = 'ModelQuality'
# # EXP_DIR_ASSIGNMENTS = 'Assignments'
# # EXP_DIR_DELTA = 'Delta'
# # EXP_DIR_IOPS = 'IoPs'
EXP_DIR_MODEL_QUALITY = 'ModelQuality'
EXP_DIR_ASSIGNMENTS = 'Assignments'
EXP_DIR_DELTA = 'Delta'
EXP_DIR_IOPS = 'IoPs'

EXP_DIR_DELTA_SELF = 'DeltaSelf'
EXP_DIR_DELTA_REFERENCE = 'DeltaReference'

# EXP_FILE_NAME_DELTA_ENSEMBLE = 'ensemble'
# EXP_FILE_NAME_DELTA_BASE = 'base'
# EXP_FILE_NAME_MODEL_QUALITY_ENSEMBLE = 'ensemble'
# EXP_FILE_NAME_MODEL_QUALITY_BASE = 'base'
# EXP_FILE_NAME_ASSIGNMENT_ENSEMBLE = 'ensemble'

# EXP_PIPELINE_NAME_PLAIN = 'plain'

EXP_IOP_DIR_AGGREGATED = 'Aggregated'
EXP_IOP_DIR_INDIVIDUAL_IOP = 'IoPs'
EXP_IOP_DIR_RISK = 'Risk'
EXP_IOP_DIR_FIGURES = 'Plots'
EXP_IOP_DIR_FIGURES_HTML = 'HTML'
EXP_IOP_DIR_FIGURES_PNG = 'PNG'


@dataclasses.dataclass
class ExpInfo:
    perc_points: float
    perc_features: float
    pipeline_name: str = dataclasses.field(default=const.MONOLITHIC_VANILLA_PIPELINE_NAME)

    #
    # def for_base(self, perc_points: float, perc_features: float) -> "ExpInfo":
    #     return ExpInfo(p)

    def to_series(self) -> pd.Series:
        return pd.Series([self.pipeline_name, self.perc_points, self.perc_features],
                         index=const.INFO_KEY_LIST)

    def prepend_to(self, target: typing.Union[pd.Series, pd.DataFrame]):
        if isinstance(target, pd.Series):
            return pd.concat([self.to_series(), target])

        add_info_to_df(info=self, df=target)
        return target

    def mini_str(self) -> str:
        return f'{self.perc_points}_{self.perc_features}'


def add_info_to_df(df: pd.DataFrame, pipeline_name: typing.Optional[str | typing.Iterable[str]] = None,
                   perc_data_points: typing.Optional[float | typing.Iterable[float]] = None,
                   perc_features: typing.Optional[float | typing.Iterable[float]] = None,
                   info: typing.Optional[ExpInfo] = None):
    if info is None and (pipeline_name is None or perc_data_points is None or perc_features is None):
        raise ValueError('info must be not None or pipeline_name and perc_data_points and perc_features '
                         'must be not None')
    if info is not None and pipeline_name is not None and perc_data_points is not None and perc_features is not None:
        raise ValueError('Either info or pipeline_name and perc_data_points and perc_features must be None')
    # assert info is not None
    pipeline_name = pipeline_name if pipeline_name is not None else info.pipeline_name
    perc_data_points = perc_data_points if perc_data_points is not None else info.perc_points
    perc_features = perc_features if perc_features is not None else info.perc_features
    df.insert(loc=0, column=const.KEY_PIPELINE_NAME, value=pipeline_name)
    df.insert(loc=1, column=const.KEY_PERC_DATA_POINTS, value=perc_data_points)
    df.insert(loc=2, column=const.KEY_PERC_FEATURES, value=perc_features)
    return df


# def df_mean_and_std_named(df: pd.DataFrame):
#     # re-add the pipeline since is removed during avg and std
#     pipeline_name = df[KEY_PIPELINE_NAME][0]
#f
#     merged = df_mean_and_std(df)
#     merged[KEY_PIPELINE_NAME] =º pipeline_name
#
#     return merged


def percentage(y_true, y_pred, how: typing.Literal["estimated", "real"]):
    if how == "estimated":
        to_be_used = y_pred
    else:
        to_be_used = y_true
    return np.count_nonzero(to_be_used)/len(to_be_used)


percentage_estimated = functools.partial(percentage, how='estimated')
percentage_real = functools.partial(percentage, how='real')

# metrics used to evaluate risk quality
RISK_METRIC_NAME_RISK_COUNT_ESTIMATED = 'Risk_Count_Est'
RISK_METRIC_NAME_RISK_COUNT_REAL = 'Risk_Count_Real'
RISK_METRIC_FUNC_COUNT = percentage
RISK_METRICS = METRICS + [(RISK_METRIC_NAME_RISK_COUNT_ESTIMATED, percentage_estimated),
                          (RISK_METRIC_NAME_RISK_COUNT_REAL, percentage_real)]

RISK_METRICS_FUNC = METRICS_FUNC + [RISK_METRIC_NAME_RISK_COUNT_ESTIMATED, RISK_METRIC_NAME_RISK_COUNT_REAL]


@dataclasses.dataclass
class AbstractExportConfig(abc.ABC):
    exists_ok: bool = dataclasses.field(default=False)


@dataclasses.dataclass
class AbstractExportConfigWithDirectory(AbstractExportConfig, abc.ABC):
    # exists_ok: bool = dataclasses.field(default=False)
    # export_also_non_aggregated: bool = dataclasses.field(default=False)
    base_directory: typing.Optional[str] = dataclasses.field(default=None)


def extract_and_evaluate_risk(p: pipe.ExtPipeline, poisoning_info: np.ndarray,
                              ) -> typing.Tuple[pd.Series, typing.Optional[pd.Series]]:
    """
    All-in-one function that retrieves the quality of the computed risk on the given pipeline,
    using step `p.pre_assignment_idx`.
    :param p:
    :param poisoning_info:
    :return:
    """

    try:
        risk_values = p.output_kept[p.pre_assignment_idx].X
    except KeyError:
        # in case pre_assignment_idx is not specified.
        risk_values = None
    thresholds = p.risk_binarization_thresholds if p.risk_binarization_thresholds is not None else [1]

    # binarized_risk = None
    if risk_values is not None:
        # we also have to evaluate the correctness of risk values.
        # since these values may be different from {0, 1},
        # we perform a variable binarization, so that we retrieve multiple values,
        # according to the pipeline configuration.
        # just the first threshold because we manage only one and that's it.
        binarized_risk = binarize_risk(risk_values=risk_values, binarization_threshold=thresholds[0])
    else:
        # TODO make sure it's correct
        binarized_risk = binarize_risk(risk_values=poisoning_info.astype(int), binarization_threshold=thresholds[0])

    return risk_quality(risk_values=binarized_risk, y_true=poisoning_info), binarized_risk

def binarize_risk(risk_values,  binarization_threshold: float):
    return np.where(risk_values >= binarization_threshold, 1, 0)


def risk_quality(risk_values: typing.Optional[np.ndarray],
                 y_true) -> pd.Series:
    # qualities = []
    #
    # for i, binarization_threshold in enumerate(binarization_thresholds):
    #     if risk_values is None:
    #         # if risk is not available just use nan but of the correct shape.
    #         collected_metrics = [np.nan for _ in RISK_METRICS_FUNC]
    #     else:
    #         binarized_risk = np.where(risk_values >= binarization_threshold, 1, 0)
    #         kwargs = {'zero_division': 1.0}
    #         collected_metrics = [metric_method(y_true=y_true, y_pred=binarized_risk,
    #                                            **kwargs if metric_name in [METRIC_NAME_RECALL, METRIC_NAME_PRECISION]
    #                                            else {})
    #                              for metric_name, metric_method in RISK_METRICS]
    #     index = [f'T{i}_{metric_name}({KEY_RISK_QUALITY})' for metric_name in RISK_METRICS]
    #     quality = pd.Series(data=collected_metrics, index=index)
    #     qualities.append(quality)

    binarized_risk_values = risk_values

    if binarized_risk_values is None:
        # if risk is not available just use nan but of the correct shape.
        collected_metrics = [np.nan for _ in RISK_METRICS_FUNC]
    else:
        kwargs = {'zero_division': 1.0}
        collected_metrics = [metric_method(y_true=y_true, y_pred=binarized_risk_values,
                                           **kwargs if metric_name in [const.METRIC_NAME_RECALL, const.METRIC_NAME_PRECISION]
                                           else {})
                             for metric_name, metric_method in RISK_METRICS]
    # index = [f'T{i}_{metric_name}({KEY_RISK_QUALITY})' for metric_name in RISK_METRICS]
    quality = pd.Series(data=collected_metrics,
                        index=[f'{metric_name}({KEY_RISK_QUALITY})' for metric_name, _ in RISK_METRICS])
    # qualities.append(quality)

    # # if risk is None, return a series of the correct shape full of np.nan
    # if risk_values is None:
    #     for i, binarization_threshold in enumerate(binarization_thresholds):
    #
    #
    # for i, binarization_threshold in enumerate(binarization_thresholds):
    #     # we retrieve the quality on an individual threshold.
    #     binarized_risk = np.where(risk_values >= binarization_threshold, 1, 0)
    #     collected_metrics = [metric_method(y_true=y_true, y_pred=binarized_risk) for metric_method in METRIC_FUNC]
    #     quality = pd.Series(data=collected_metrics,
    #                         index=[f'T{i}_{metric_name}({KEY_RISK_QUALITY})'
    #                                for metric_name in METRICS_NAME])
    #     qualities.append(quality)
    # and we concat the different pd.Series
    # return pd.concat(qualities)
    return quality


def check_unique_pipeline_names(pipelines: typing.Sequence[pipe.ExtPipeline]):
    if len(set(p.name for p in pipelines)) != len(pipelines):
        raise ValueError('There are some pipelines with non-unique names. '
                         f'Unique names: {set(p.name for p in pipelines)}, '
                         f'all names: {[p.name for p in pipelines]}')
