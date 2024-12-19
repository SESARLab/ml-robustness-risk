import abc
import copy
import dataclasses
import enum
import os
import typing
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn import base as sk_base
import xarray as xr

import assignments
import models
import pipe
import utils
from . import base
import const, utils_exp_post


def evaluate_metrics_to_series(y_pred, y_test, info: base.ExpInfo):
    collected_metrics = [metric_method(y_true=y_test, y_pred=y_pred) for metric_method in base.METRICS_FUNC]
    return info.prepend_to(pd.Series(data=collected_metrics,
                                                    index=[metric_name for metric_name in base.METRICS_NAME]))


class TestSetType(enum.Enum):
    CLEAN_TEST_SET = 'CLEAN_TEST_SET'
    CLEAN_TRAINING_SET = 'CLEAN_TRAINING_SET'

    def directory_name(self) -> str:
        if self == TestSetType.CLEAN_TEST_SET:
            return base.EXP_DIR_TEST_SET_CLEAN
        return base.EXP_DIR_TRAINING_SET_CLEAN

    def prefix(self) -> str:
        if self == TestSetType.CLEAN_TEST_SET:
            return base.EXPORT_NAME_PREFIX_TEST_SET_TYPE_TEST_SET_CLEAN
        return base.EXPORT_NAME_PREFIX_TEST_SET_TYPE_TRAINING_SET_CLEAN


@dataclasses.dataclass
class TrainSingleOutputSingleTestSetMonolithic:
    """
    Models the result of an individual execution and evaluation of monolithic model over an individual test set.

    Attributes:
        test_set_type: TestSetType
        model_quality: pd.Series
    """
    # the type of test set used for the evaluation.
    test_set_type: TestSetType
    model_quality: pd.Series

    @staticmethod
    def from_results(y_pred, y_test, info: base.ExpInfo, test_set_type: TestSetType):
        model_quality = evaluate_metrics_to_series(y_pred=y_pred, y_test=y_test, info=info)
        return TrainSingleOutputSingleTestSetMonolithic(test_set_type=test_set_type, model_quality=model_quality)


@dataclasses.dataclass
class TrainSingleOutputMonolithic:
    """
    Models the result of an execution and evaluation of monolithic model over a set of test sets.
    Each item in the list has been retrieved on a different test sets.

    Attributes:
        info: base.ExpInfo
        results: typing.List[TrainSingleOutputSingleTestSetMonolithic]
    """
    info: base.ExpInfo
    results: typing.List[TrainSingleOutputSingleTestSetMonolithic]


@dataclasses.dataclass
class TrainSingleOutputMonolithicWithRep:
    """
    The output of the monolithic model training on a single percentage of poisoning multiple times.
    It is a dictionary mapping the test set type with the results.

    Attributes:
        model_quality: typing.Dict[TestSetType, pd.Series]
        info: base.ExpInfo
    """
    model_quality: typing.Dict[TestSetType, pd.Series]
    info: base.ExpInfo

    @staticmethod
    def from_reps(reps: typing.Sequence[TrainSingleOutputMonolithic]) -> "TrainSingleOutputMonolithicWithRep":

        # we need to aggregate over the test set types.
        # test set type -> result as df
        raw_results = {t: [] for t in TestSetType}
        for rep in reps:
            for result in rep.results:
                raw_results[result.test_set_type].append(result.model_quality)

        # NOTE: all these results already contain information about the percentage of poisoning.
        final_results = {}
        for test_set_type, results in raw_results.items():
            df = pd.DataFrame(results)
            final_results[test_set_type] = utils_exp_post.df_mean_and_std_drop_and_add_info(df, info=reps[0].info)

        return TrainSingleOutputMonolithicWithRep(model_quality=final_results, info=reps[0].info)


def get_last_step_from_estimator(estimator: models.EnsembleWithAssignmentPipeline):
    p: pipe.ExtPipeline = estimator.data_point_assignment

    last_step: assignments.AbstractAssignment = p.steps[-1].step
    # these things should never happen because they have already checked, but let's check it
    # just in case.
    if not isinstance(last_step, assignments.AbstractAssignment):
        raise ValueError('The last step of the pipeline must be an instance of '
                         f'\'assignments.AbstractAssignment\'. Got: {type(last_step)}')
    return p, last_step


@dataclasses.dataclass
class TrainSingleOutputSingleTestSetEnsemble:
    """
    The output of an ensemble-based training (single execution) evaluated on an individual test set.

    Attributes:
        model_quality: pd.Series

        assignment_quality: pd.Series

        info: base.ExpInfo

        test_set_type: TestSetType
    """
    # it includes the pipeline name percentage of poisoning
    model_quality: pd.Series
    # it includes the pipeline name percentage of poisoning
    assignment_quality: pd.Series
    info: base.ExpInfo
    # the type of test set used for the evaluation.
    test_set_type: TestSetType

    @staticmethod
    def from_results(*,
                     estimator: models.EnsembleWithAssignmentPipeline,
                     X_train,
                     test_set_type: TestSetType,
                     y_pred, y_test,
                     info: base.ExpInfo,  # X_y: pd.DataFrame,
                     hard_count: np.ndarray,
                     poisoning_info: np.ndarray) -> "TrainSingleOutputSingleTestSetEnsemble":
        """
        :param test_set_type:
        :param X_train:
        :param estimator:
        :param y_pred:
        :param y_test:
        :param info:
        :param hard_count: the count used to retrieve discordance/concordance in the predictions.
        :param poisoning_info:
        :return:
        """
        model_quality = evaluate_metrics_to_series(y_pred=y_pred, y_test=y_test, info=info)

        # now we retrieve the quality of the assignment.
        # What we can do so far is only the discordance.
        majority_count_avg = np.average(hard_count)
        majority_count_std = np.std(hard_count)
        # this is an individual number telling the number of discordance(s)
        # happened within this prediction on X.
        # let's assume that the count is as follows:
        # array([1.        , 0.66666667, 1.        , 0.66666667, 0.66666667,
        #        1.        , 0.66666667, 1.        , 1.        , 1.        ,
        #        0.66666667, 0.66666667, 1.        , 0.66666667, 1.        ,
        #        0.66666667, 1.        , 1.        , 1.        , 0.66666667,
        #        1.        , 1.        , 0.66666667, 0.66666667, 1.        ])
        # when 1 = perfect accordance, when 0.67 it means that 2/3 agrees, and so on.
        # Now, we are only interested in counting the number of discordance(s) regardless
        # the entity (i.e., an eventual discordance of 0.5 is treated the same as 0.67).
        # Then we normalize this count.
        majority_count_discordant = np.count_nonzero(hard_count != 1) / len(hard_count)
        majority_s = pd.Series([majority_count_avg, majority_count_std, majority_count_discordant],
                               index=[base.KEY_MAJORITY_COUNT_AVG, base.KEY_MAJORITY_COUNT_STD,
                                              base.KEY_COUNT_N_DISCORDANT])

        p, last_step = get_last_step_from_estimator(estimator=estimator)

        assignment_quality = last_step.get_custom_quality_metrics(X_train=X_train, y_pred=y_pred, y_test=y_test,
                                                                  risk_values=None, with_risk=False,
                                                                  poisoning_idx=poisoning_info)
        if assignment_quality is None:
            assignment_quality = pd.Series()

        assignment_quality = pd.concat([assignment_quality, majority_s])

        # finally we put together assignment and risk quality in a single pd.Series
        assignment_quality = info.prepend_to(assignment_quality)

        return TrainSingleOutputSingleTestSetEnsemble(model_quality=model_quality,
                                                      assignment_quality=assignment_quality, info=info,
                                                      test_set_type=test_set_type)

@dataclasses.dataclass
class TrainSingleOutputEnsemble:
    """
    Models the result of an execution and evaluation of the ensemble over a set of test sets.

    Attributes:
        info: base.ExpInfo

        results: typing.List[TrainSingleOutputSingleTestSetEnsemble]
    """
    info: base.ExpInfo
    results: typing.List[TrainSingleOutputSingleTestSetEnsemble]


@dataclasses.dataclass
class TrainSingleOutputEnsembleWithRep:
    """
    The output of an ensemble model training on a single percentage of poisoning multiple times.
    It is a dictionary mapping the test set type with the results.

    Attributes:
        info : base.ExpInfo

        model_quality : typing.Dict[TestSetType, pd.Series]

        assignment_quality : typing.Dict[TestSetType, pd.Series]
    """
    # model_quality: pd.Series
    model_quality: typing.Dict[TestSetType, pd.Series]
    assignment_quality: typing.Dict[TestSetType, pd.Series]
    # assignment_quality: pd.Series

    info: base.ExpInfo

    @staticmethod
    def from_reps(reps: typing.Sequence[TrainSingleOutputEnsemble]) -> "TrainSingleOutputEnsembleWithRep":
        """
        :param reps: sequence of executions (evaluated on different test sets)
        :return:
        """
        model_quality = {t: [] for t in TestSetType}
        assignment_quality = {t: [] for t in TestSetType}

        for rep in reps:
            for result in rep.results:
                model_quality[result.test_set_type].append(result.model_quality)
                assignment_quality[result.test_set_type].append(result.assignment_quality)

        final_model_quality = {}
        final_assignment_quality = {}
        for test_set_type in TestSetType:
            final_model_quality[test_set_type] = utils_exp_post.df_mean_and_std_drop_and_add_info(
                pd.DataFrame(model_quality[test_set_type]), info=reps[0].info)
            # note that even if the pipelines have different columns here (e.g., those specific
            # of each assignment function) they do not cause an issue. For instance:
            # s1 = pd.Series([1, 2, 3], index=list('abc'))
            # s2 = pd.Series([1, 2, 3], index=list('abc'))
            # s3 = pd.Series([1, 2, 4], index=list('abd'))
            # pd.DataFrame([s1, s2, s3])
            #      a    b    c    d
            # 0  1.0  2.0  3.0  NaN
            # 1  2.0  4.0  6.0  NaN
            # 2  1.0  2.0  NaN  4.0
            final_assignment_quality[test_set_type] = utils_exp_post.df_mean_and_std_drop_and_add_info(
                pd.DataFrame(assignment_quality[test_set_type]), info=reps[0].info)
        #
        info = reps[0].info

        return TrainSingleOutputEnsembleWithRep(model_quality=final_model_quality,
                                                assignment_quality=final_assignment_quality, info=info)


T_RepOutput = typing.TypeVar('T_RepOutput', TrainSingleOutputEnsembleWithRep, TrainSingleOutputMonolithicWithRep)


@dataclasses.dataclass
class CleanPoisonedOutputPair(typing.Generic[T_RepOutput]):
    """
    Just a simple container of different executions (different percentages of poisoning) over the same pipeline.
    """
    clean: T_RepOutput
    poisoned: typing.List[T_RepOutput]
    pipeline_name: str


def columns_to_consider_in_delta(overall_df: pd.DataFrame) -> typing.List[str]:
    columns = list(set(overall_df.columns) - const.INFO_KEYS_SET)
    columns = [col for col in columns if not col.startswith(const.PREFIX_STD)]
    return columns


@dataclasses.dataclass
class AnalyzedOutputPairMonolithic:
    # the delta against the *same* (monolithic) model trained the clean dataset (i.e., current_result - result_clean)
    delta_self: typing.Dict[TestSetType, pd.DataFrame]
    # "Raw" quality of the model
    model_quality: typing.Dict[TestSetType, pd.DataFrame]

    @staticmethod
    def columns_to_consider_in_delta(overall_df: pd.DataFrame) -> typing.List[str]:
        return columns_to_consider_in_delta(overall_df=overall_df)

    @staticmethod
    def from_results(result: CleanPoisonedOutputPair, **kwargs):
        # result_.poisoned is a list of TrainSingleOutputMonolithicWithRep

        # we retrieve two deltas
        # - the first one is the same we used in the past, i.e., with respect
        #   to the same model but on the clean dataset.
        # - the second is retrieved *from the base model using the same percentage of poisoning*.

        # We do this operation by iterating over the test set types.
        raw_results: typing.Dict[TestSetType, typing.List[pd.Series]] = {t: [] for t in TestSetType}

        for single_result in result.poisoned:
            single_result: TrainSingleOutputMonolithicWithRep = single_result
            # raw_results[single_result.test_set_type].append(single_result.model_quality)
            for t in single_result.model_quality.keys():
                raw_results[t].append(single_result.model_quality[t])

        # dict where we put the final result. Index is the test set type.
        model_quality_accu = {}
        delta_self_accu = {}

        for test_set_type, overall_result in raw_results.items():

            # here we are putting the result of poisoning one after the other i.e.,
            # poisoning_perc    avg(recall)  avg(acc)   std(recall)
            # 5                 0.7             0.8     ...
            # 10                0.69            0.79    ...
            overall_df = pd.DataFrame(overall_result)

            # now, not all columns in our dataframe are relevant.
            # So we build a list of columns that are relevant.
            # For instance, we don't retrieve the delta on std but only on avg.
            columns_to_consider_in_delta_ = AnalyzedOutputPairMonolithic.columns_to_consider_in_delta(
                overall_df=overall_df)

            # here we work with poisoned results contained in a DataFrame as:
            # index         avg(recall)       avg(acc)   std(recall)
            # 0                 0.7            0.8          ...
            # 1                0.69            0.79         ...

            # then, the clean result are contained in a Series as:
            # avg(recall)   1
            # avg(acc)      0.99

            # so we subtract the Series from the DataFrame: the operation is executed on all columns, e.g.,
            # df[0] - series
            # df[1] - series

            # this is the first delta, i.e., the traditional one, retrieved against "itself"
            delta_self = overall_df[columns_to_consider_in_delta_] - result.clean.model_quality[test_set_type][
                columns_to_consider_in_delta_]

            delta_self = base.add_info_to_df(df=delta_self, pipeline_name=overall_df[const.KEY_PIPELINE_NAME],
                                        perc_data_points=overall_df[const.KEY_PERC_DATA_POINTS],
                                        perc_features=overall_df[const.KEY_PERC_FEATURES])

            # now each column must also be renamed.
            # Note that the prefix that we add is the same because they will be saved
            # in different files so there's no risk of conflicts.
            delta_self = delta_self.rename(
                lambda col: f'{const.PREFIX_DELTA}({col})' if col in columns_to_consider_in_delta_ else col,
                axis='columns')
            # we append the results on the clean dataset at the beginning. To do that, we wrap
            # it in a pd.DataFrame, and then use .T because a plain pd.Series looks like a "column"
            # array (while we want a row instead).
            # Then we reset the index otherwise we have two 0s.
            model_quality = pd.concat([pd.DataFrame(result.clean.model_quality[test_set_type]).T, overall_df]
                                      ).reset_index(drop=True)
            delta_self_accu[test_set_type] = delta_self
            model_quality_accu[test_set_type] = model_quality

        return AnalyzedOutputPairMonolithic(model_quality=model_quality_accu, delta_self=delta_self_accu)


def compute_delta_ref(first: CleanPoisonedOutputPair, baseline: CleanPoisonedOutputPair
                      ) -> typing.Dict[TestSetType, pd.DataFrame]:
    delta_ref_accu = {}

    for test_set_type in TestSetType:

        # accumulate all the results of the "first" in a unique pd.DataFrame
        first_with_clean = pd.DataFrame([first.clean.model_quality[test_set_type]] +
                                        [r.model_quality[test_set_type] for r in first.poisoned])

        # and do the same for the baseline.
        baseline_with_clean = pd.DataFrame([baseline.clean.model_quality[test_set_type]] +
                                           [r.model_quality[test_set_type] for r in baseline.poisoned])

        # now, not all columns are relevant.
        # which columns do we have to use?
        columns_to_consider_in_delta_ = columns_to_consider_in_delta(overall_df=baseline_with_clean)

        delta_ref = first_with_clean[columns_to_consider_in_delta_] - baseline_with_clean[columns_to_consider_in_delta_]
        delta_ref = base.add_info_to_df(df=delta_ref, pipeline_name=first_with_clean[const.KEY_PIPELINE_NAME],
                                        perc_data_points=first_with_clean[const.KEY_PERC_DATA_POINTS],
                                        perc_features=first_with_clean[const.KEY_PERC_FEATURES])
        # now, rename the columns adding the correct prefix.
        delta_ref = delta_ref.rename(
                lambda col: f'{const.PREFIX_DELTA}({col})' if col in columns_to_consider_in_delta_ else col,
                axis='columns')
        delta_ref_accu[test_set_type] = delta_ref
    return delta_ref_accu


@dataclasses.dataclass
class AnalyzedOutputPairMonolithicWithOracle:
    """
    Attributes
    ----------
    model_quality_vanilla: typing.Dict[TestSetType, pd.DataFrame] quality of the vanilla monolithic model

    delta_self_vanilla: typing.Dict[TestSetType, pd.DataFrame] delta of the vanilla monolithic model,
        retrieved as vanilla(poisoned) - vanilla(clean)

    model_quality_oracled: typing.Dict[TestSetType, pd.DataFrame] quality of the monolithic model where
        poisoned data points are filtered out

    delta_self_oracled: typing.Dict[TestSetType, pd.DataFrame] delta of the oracle monolithic model,
        retrieved as oracle(poisoned) - oracle(clean). Note that strictly speaking the oracle is never poisoned,
        so when we say poisoned we either refer to *the monolithic model where all the poisoned data points are
        removed from the training set*. On the other hand, `oracle(clean)≠ is the same as `vanilla(clean)`.

    delta_ref_oracled: typing.Dict[TestSetType, pd.DataFrame],
        retrieved as oracle(poisoned) - vanilla(poisoned) with the same percentage of poisoning.
        It represents the loss against an (kinda) *ideal* defense.

    """
    model_quality_vanilla: typing.Dict[TestSetType, pd.DataFrame]
    delta_self_vanilla: typing.Dict[TestSetType, pd.DataFrame]
    model_quality_oracled: typing.Dict[TestSetType, pd.DataFrame]
    delta_self_oracled: typing.Dict[TestSetType, pd.DataFrame]
    # this is delta retrieved as oracle - monolithic with the same percentage of poisoning.
    delta_ref_oracled: typing.Dict[TestSetType, pd.DataFrame]

    @staticmethod
    def from_results(result_vanilla: CleanPoisonedOutputPair, result_oracled: CleanPoisonedOutputPair, **kwargs
                     ) -> "AnalyzedOutputPairMonolithicWithOracle":
        # basically we are ready to compute most of the results we need.
        analyzed_results_vanilla = AnalyzedOutputPairMonolithic.from_results(result=result_vanilla)
        analyzed_results_oracle = AnalyzedOutputPairMonolithic.from_results(result=result_oracled)

        # now we have everything we need but the delta_ref.
        delta_ref_oracled = compute_delta_ref(first=result_oracled, baseline=result_vanilla)
        # and we are ready to return.
        return AnalyzedOutputPairMonolithicWithOracle(model_quality_vanilla=analyzed_results_vanilla.model_quality,
                                                      delta_self_vanilla=analyzed_results_vanilla.delta_self,
                                                      model_quality_oracled=analyzed_results_oracle.model_quality,
                                                      delta_self_oracled=delta_ref_oracled,
                                                      delta_ref_oracled=delta_ref_oracled,)


@dataclasses.dataclass
class AnalyzedOutputPairEnsemble:
    """
    Result of training/evaluation of an ensemble (i.e., same pipeline).

    **NOTE**: I preferred not to extend ~AnalyzedOutputPairMonolithicWithOracle even if some attributes
    are in common. It just creates additional confusion

    Attributes
    ----------
    model_quality: typing.Dict[TestSetType, pd.DataFrame] quality of the ensemble

    delta_self: typing.Dict[TestSetType, pd.DataFrame] delta of the ensemble,
        retrieved as ensemble(poisoned) - ensemble(clean)

    delta_ref_monolithic_vanilla: typing.Dict[TestSetType, pd.DataFrame] delta of the ensemble against the
        vanilla monolithic model. Retrieved as ensemble(poisoned) - vanilla(monolithic(poisoned)).

    delta_ref_monolithic_oracled: typing.Dict[TestSetType, pd.DataFrame] delta of the ensemble against the
        oracle monolithic model. Retrieved as ensemble(poisoned) - oracle(monolithic(poisoned)).

    assignment_quality: typing.Dict[TestSetType, pd.DataFrame]
    """
    model_quality: typing.Dict[TestSetType, pd.DataFrame]
    delta_self: typing.Dict[TestSetType, pd.DataFrame]
    delta_ref_monolithic_vanilla: typing.Dict[TestSetType, pd.DataFrame]
    delta_ref_monolithic_oracled: typing.Dict[TestSetType, pd.DataFrame]
    # then we will delta_monolithic_oracle
    assignment_quality: typing.Dict[TestSetType, pd.DataFrame]

    @staticmethod
    def from_results(result: CleanPoisonedOutputPair[TrainSingleOutputEnsembleWithRep],
                     reference_monolithic_model_vanilla: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep],
                     reference_monolithic_model_oracled: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep]):

        # delta against self can be retrieved by calling this method.
        preliminary_results = AnalyzedOutputPairMonolithic.from_results(result=result)

        # delta_ref_monolithic_accu = {}
        assignment_quality_accu = {}

        for test_set_type in TestSetType:
            assignment_quality_s = []
            for single_poisoned_result in result.poisoned:
                assignment_quality_s.append(single_poisoned_result.assignment_quality[test_set_type])
            assignment_quality_s = [result.clean.assignment_quality[test_set_type]] + assignment_quality_s
            # and now we calculate the usual mean and std and that's it.
            assignment_quality_accu[test_set_type] = pd.DataFrame(assignment_quality_s).reset_index().drop(
                'index', axis='columns')

        delta_ref_monolithic_vanilla_accu = compute_delta_ref(first=result, baseline=reference_monolithic_model_vanilla)
        delta_ref_monolithic_oracled_accu = compute_delta_ref(first=result, baseline=reference_monolithic_model_oracled)

        return AnalyzedOutputPairEnsemble(delta_ref_monolithic_vanilla=delta_ref_monolithic_vanilla_accu,
                                          delta_ref_monolithic_oracled=delta_ref_monolithic_oracled_accu,
                                          delta_self=preliminary_results.delta_self,
                                          model_quality=preliminary_results.model_quality,
                                          assignment_quality=assignment_quality_accu)


@dataclasses.dataclass
class AnalyzedResultsEnsembleCommon:
    # results on the monolithic model (test set type -> results)
    monolithic_model_quality_vanilla: typing.Dict[TestSetType, pd.DataFrame]
    # delta against self (old) (test set type -> results)
    monolithic_delta_self_vanilla: typing.Dict[TestSetType, pd.DataFrame]

    monolithic_model_quality_oracled: typing.Dict[TestSetType, pd.DataFrame]
    monolithic_delta_self_oracled: typing.Dict[TestSetType, pd.DataFrame]
    monolithic_delta_ref_vanilla_oracled: typing.Dict[TestSetType, pd.DataFrame]

    # results on the ensemble (the index is the pipeline name, the value is [test set type, result])
    ensemble_model_quality: typing.Dict[str, typing.Dict[TestSetType, pd.DataFrame]]

    # delta against vanilla monolithic model (new)
    ensemble_delta_ref_vanilla: typing.Dict[str, typing.Dict[TestSetType, pd.DataFrame]]
    # delta against oracled monolithic model (new)
    ensemble_delta_ref_oracled: typing.Dict[str, typing.Dict[TestSetType, pd.DataFrame]]

    # delta against self (old, TSUSC style)
    ensemble_delta_self: typing.Dict[str, typing.Dict[TestSetType, pd.DataFrame]]

    # assignment/risk quality
    assignment_quality: typing.Dict[str, typing.Dict[TestSetType, pd.DataFrame]]

    @staticmethod
    def from_results(results_monolithic_vanilla: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep],
                     results_monolithic_oracled: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep],
                     results_ensemble: typing.List[CleanPoisonedOutputPair[TrainSingleOutputEnsembleWithRep]],
                     ) -> "AnalyzedResultsEnsembleCommon":
        # results_monolithic_analyzed = AnalyzedOutputPairMonolithic.from_results(result=results_monolithic_vanilla)
        results_monolithic_analyzed = AnalyzedOutputPairMonolithicWithOracle.from_results(
            result_vanilla=results_monolithic_vanilla, result_oracled=results_monolithic_oracled,)

        model_quality_ensemble_accu = {}
        delta_ensemble_ref_monolithic_vanilla_accu = {}
        delta_ensemble_ref_monolithic_oracled_accu = {}
        delta_ensemble_self_accu = {}
        assignment_quality_ensemble_accu = {}

        for result_ensemble in results_ensemble:
            result_ensemble_analyzed = AnalyzedOutputPairEnsemble.from_results(
                result=result_ensemble, reference_monolithic_model_vanilla=results_monolithic_vanilla,
                reference_monolithic_model_oracled=results_monolithic_oracled)

            model_quality_ensemble_accu[result_ensemble.pipeline_name] = result_ensemble_analyzed.model_quality
            delta_ensemble_ref_monolithic_vanilla_accu[
                result_ensemble.pipeline_name] = result_ensemble_analyzed.delta_ref_monolithic_vanilla
            delta_ensemble_ref_monolithic_oracled_accu[
                result_ensemble.pipeline_name] = result_ensemble_analyzed.delta_ref_monolithic_oracled
            delta_ensemble_self_accu[result_ensemble.pipeline_name] = result_ensemble_analyzed.delta_self
            assignment_quality_ensemble_accu[result_ensemble.pipeline_name] = result_ensemble_analyzed.assignment_quality

        return AnalyzedResultsEnsembleCommon(
            monolithic_model_quality_vanilla=results_monolithic_analyzed.model_quality_vanilla,
            monolithic_model_quality_oracled=results_monolithic_analyzed.model_quality_oracled,
            monolithic_delta_self_vanilla=results_monolithic_analyzed.delta_self_vanilla,
            monolithic_delta_self_oracled=results_monolithic_analyzed.delta_self_oracled,
            monolithic_delta_ref_vanilla_oracled=results_monolithic_analyzed.delta_ref_oracled,
            ensemble_model_quality=model_quality_ensemble_accu,
            ensemble_delta_self=delta_ensemble_self_accu,
            ensemble_delta_ref_vanilla=delta_ensemble_ref_monolithic_vanilla_accu,
            ensemble_delta_ref_oracled=delta_ensemble_ref_monolithic_oracled_accu,
            assignment_quality=assignment_quality_ensemble_accu)

    def export(self, config: base.AbstractExportConfigWithDirectory):
        if config.base_directory is None:
            return
        # now we create one directory for each "result type".
        # We begin with the merged results directory, that contains
        # the join of delta, quality and so for easy vision.
        dir_merged = os.path.join(config.base_directory, base.EXP_DIR_MERGED)
        dir_delta_self = os.path.join(config.base_directory, base.EXP_DIR_DELTA_SELF)
        dir_delta_reference = os.path.join(config.base_directory, base.EXP_DIR_DELTA_REFERENCE)
        dir_model_quality = os.path.join(config.base_directory, base.EXP_DIR_MODEL_QUALITY)
        dir_assignments = os.path.join(config.base_directory, base.EXP_DIR_ASSIGNMENTS)
        # dir_iops = os.path.join(config.base_directory, base.EXP_DIR_IOPS)
        dirs = [dir_merged, dir_delta_self, dir_delta_reference, dir_model_quality, dir_assignments]
        for dir_to_create in dirs:
            os.makedirs(dir_to_create, exist_ok=config.exists_ok)

        # let's begin by MERGING things (i.e., results over different pipelines
        # in a unique pd.DataFrame).
        #
        # for the file ((FILE_NAME_EXPORT_MODEL_ENSEMBLE, self.ensemble_model_quality))
        # containing the aggregated accuracy, prec, etc.,
        # values of all pipelines, we also add one column containing the results
        # of the training on the monolithic models
        #
        # so we just update the source, instead of only `self.ensemble_model_quality`,
        # we also add the results on base. Note that there's nothing to be done,
        # since the format of `self.base_model_quality` is the same of any element in `self.ensemble_model_quality`.
        src_for_model_quality = {const.MONOLITHIC_VANILLA_PIPELINE_NAME: self.monolithic_model_quality_vanilla,
                                 const.MONOLITHIC_ORACLED_PIPELINE_NAME: self.monolithic_model_quality_oracled,
                                 **self.ensemble_model_quality}

        for target_name, src in [(base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_SELF, self.ensemble_delta_self),
                                 (base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_VANILLA, self.ensemble_delta_ref_vanilla),
                                 (base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_ORACLED, self.ensemble_delta_ref_oracled),
                                 # (FILE_NAME_EXPORT_MODEL_ENSEMBLE, self.ensemble_model_quality),
                                 (base.FILE_NAME_EXPORT_ENSEMBLE_QUALITY, src_for_model_quality),
                                 (base.FILE_NAME_EXPORT_ENSEMBLE_ASSIGNMENT, self.assignment_quality),
            # for the monolithic models, we need to "wrap" with one key more, that is, the pipeline name.
            # this way, we match the structure of other items, which is dict of pipeline_name -> [test_set_type -> [results]]
                                 (base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, {const.MONOLITHIC_VANILLA_PIPELINE_NAME: self.monolithic_delta_self_vanilla}),
                                 (base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF, {const.MONOLITHIC_ORACLED_PIPELINE_NAME: self.monolithic_delta_self_oracled}),
                                 (base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED, {const.MONOLITHIC_ORACLED_PIPELINE_NAME: self.monolithic_delta_ref_vanilla_oracled})
                                 ]:

            # merged = merge_repeatedly_and_drop_unnecessary_columns(list(src.values()),
            #                                                                pipeline_names=list(src.keys()),
            #                                                                drop_std=True)

            to_merge = {t: dict() for t in TestSetType}

            # now, src is dict of pipeline_name -> [test_set_type -> [results]]
            for pipeline_name, results in src.items():
                for test_set_type, result in results.items():
                    to_merge[test_set_type].update({pipeline_name: result})

            for test_set_type, results in to_merge.items():
                merged = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(list(results.values()),
                                                                       pipeline_names=list(results.keys()),
                                                                       drop_std=True)

                merged.to_csv(os.path.join(dir_merged, f'{target_name}_{test_set_type.prefix()}.csv'), index=False)

        # we now export non-merged values (for the ensemble)
        for base_dir, target_name, src in [
            (dir_delta_reference, base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_VANILLA, self.ensemble_delta_ref_vanilla),
            (dir_delta_reference, base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_ORACLED, self.ensemble_delta_ref_oracled),
            (dir_delta_self, base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_SELF, self.ensemble_delta_self),
            (dir_assignments, base.FILE_NAME_EXPORT_ENSEMBLE_ASSIGNMENT, self.assignment_quality),
            (dir_model_quality, base.FILE_NAME_EXPORT_ENSEMBLE_QUALITY, self.ensemble_model_quality)]:

            for pipeline_name, result_of_pipeline in src.items():
                for test_set_type, result in result_of_pipeline.items():
                    result.to_csv(os.path.join(base_dir, f'{target_name}_{pipeline_name}_{test_set_type.prefix()}.csv'),
                                  index=False)

        # we finally export the results for the monolithic model
        for base_dir, target_name, src in [
            (dir_delta_self, base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, self.monolithic_delta_self_vanilla),
            (dir_delta_self, base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF, self.monolithic_delta_self_oracled),
            (dir_model_quality, base.FILE_NAME_EXPORT_MONO_VANILLA_QUALITY, self.monolithic_model_quality_vanilla),
            (dir_model_quality, base.FILE_NAME_EXPORT_MONO_ORACLED_QUALITY, self.monolithic_model_quality_oracled),
            (dir_delta_reference, base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED, self.monolithic_delta_ref_vanilla_oracled)
        ]:
            for test_set_type, result in src.items():
                result.to_csv(os.path.join(base_dir, f'{target_name}_{test_set_type.prefix()}.csv'), index=False)


@dataclasses.dataclass
class ExportConfigExpEnsembleCommon(base.AbstractExportConfig):
    pass


TEstimator = typing.TypeVar('TEstimator', bound=utils.EstimatorProtocol)


class AbstractTrainModelWithRepMixin(abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def get_repetitions(self) -> int:
        pass

    @abc.abstractmethod
    def get_X_y_test(self, test_set_type: TestSetType) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass

    def _train_monolithic_estimator(self, estimator, X_train, y_train, info: base.ExpInfo, poisoning_info: np.ndarray
                                    ) -> TrainSingleOutputMonolithic:
        estimator.fit(X=X_train, y=y_train)
        # y_pred = estimator_.predict(self.X_test)
        # return TrainSingleOutputMonolithic.from_results(y_pred=y_pred, y_test=self.y_test, info=info_)
        results_ = []
        # for (X_test, y_test, test_set_type) in [(self.X_test, self.y_test, TestSetType.CLEAN_TEST_SET),
        #                                         (self.X_train_clean, self.y_train_clean, TestSetType.CLEAN_TRAINING_SET)]:
        for test_set_type in TestSetType:
            X_test, y_test = self.get_X_y_test(test_set_type)
            y_pred = estimator.predict(X_test)
            result = TrainSingleOutputSingleTestSetMonolithic.from_results(
                y_test=y_test, y_pred=y_pred, info=info, test_set_type=test_set_type)
            results_.append(result)
        return TrainSingleOutputMonolithic(results=results_, info=info)

    def _callback_monolithic_vanilla(self, estimator, X_train, y_train, info: base.ExpInfo, poisoning_info: np.ndarray
                                     ) -> TrainSingleOutputMonolithic:
        return self._train_monolithic_estimator(estimator=estimator, X_train=X_train, y_train=y_train, info=info,
                                                poisoning_info=poisoning_info)

    def _callback_monolithic_oracle(self, estimator, X_train, y_train, info: base.ExpInfo, poisoning_info: np.ndarray
                                    ) -> TrainSingleOutputMonolithic:
        assert isinstance(estimator, models.EstimatorWithOracle)
        estimator.poisoning_info = poisoning_info
        return self._train_monolithic_estimator(estimator=estimator, X_train=X_train, y_train=y_train, info=info,
                                                poisoning_info=poisoning_info)

    def _callback_ensemble(self, estimator, X_train, y_train, info: base.ExpInfo, poisoning_info: np.ndarray):
        results_ = []
        #craised = False
        estimator.fit(X=X_train, y=y_train)
            #craised = True
        # for (X_test, y_test, test_set_type) in [(self.X_test, self.y_test, TestSetType.CLEAN_TEST_SET),
        #                                         (self.X_train_clean, self.y_train_clean, TestSetType.CLEAN_TRAINING_SET)]:
        for test_set_type in TestSetType:
            X_test, y_test = self.get_X_y_test(test_set_type)
            y_pred, count = estimator.hard_predictions_count(X_test)
            # if not raised:
            #     y_pred, count = estimator.hard_predictions_count(X_test)
            # else:
            #    y_pred, count = np.repeat(np.nan, len(y_test)), np.repeat(np.nan, len(y_test))
            result = TrainSingleOutputSingleTestSetEnsemble.from_results(
                estimator=estimator, X_train=X_train, y_test=y_test, y_pred=y_pred,
                test_set_type=test_set_type, info=info, hard_count=count, poisoning_info=poisoning_info)
            results_.append(result)
        return TrainSingleOutputEnsemble(info=info, results=results_)


    def train_model_with_rep(self, estimator: typing.Union[models.EnsembleWithAssignmentPipeline, TEstimator],
                             X_train: np.ndarray, y_train: np.ndarray,
                             info: base.ExpInfo, poisoning_info: np.ndarray
                             ) -> TrainSingleOutputEnsembleWithRep | TrainSingleOutputMonolithicWithRep:

        def _inner_func(estimator_, X_train_: np.ndarray, y_train_: np.ndarray,
                        info_: base.ExpInfo, poisoning_info_: np.ndarray):

            # if it is an ensemble with pipeline, then we train it differently.
            if isinstance(estimator_, models.EnsembleWithAssignmentPipeline):
                return self._callback_ensemble(estimator=estimator_, X_train=X_train_, y_train=y_train_,
                                               info=info_, poisoning_info=poisoning_info_)

            elif isinstance(estimator_, models.EstimatorWithOracle):
                return self._callback_monolithic_oracle(estimator=estimator_, X_train=X_train_, y_train=y_train_,
                                                        info=info_, poisoning_info=poisoning_info_)
            else:
                return self._callback_monolithic_vanilla(estimator=estimator_, X_train=X_train_, y_train=y_train_,
                                                         info=info_, poisoning_info=poisoning_info_)

        with joblib.Parallel(n_jobs=self.get_repetitions()) as parallel:
            results = parallel(
                joblib.delayed(_inner_func)(
                    estimator_=copy.deepcopy(estimator), X_train_=X_train, y_train_=y_train, info_=info,
                    poisoning_info_=poisoning_info
                ) for _ in range(self.get_repetitions()))
        receiver_to_call = TrainSingleOutputEnsembleWithRep if isinstance(
            estimator, models.EnsembleWithAssignmentPipeline) else TrainSingleOutputMonolithicWithRep
        return receiver_to_call.from_reps(results)


class AbstractCommonExperiment(base.AbstractExperiment, AbstractTrainModelWithRepMixin, abc.ABC):

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

        super().__init__(repetitions=repetitions, clean_dataset_attrs={},
                         poisoned_datasets=poisoned_datasets, columns=columns)

        self.X_train_clean = X_train_clean
        self.y_train_clean = y_train_clean
        self.X_test = X_test
        self.y_test = y_test

        self.monolithic_model = monolithic_model

        # here we create the two additional pipelines.

        self.pipelines = pipelines

        self.results_clean_base: TrainSingleOutputMonolithicWithRep = None
        self.results_clean_ensemble: typing.Dict[str, TrainSingleOutputEnsembleWithRep] = {}

        self.results_poisoned_base: typing.Dict[str, typing.List[TrainSingleOutputEnsembleWithRep]] = {}
        self.results_poisoned_ensemble: typing.Dict[str, typing.List[TrainSingleOutputEnsembleWithRep]] = {}

        for i, p in enumerate(pipelines):
            if p.name is None:
                raise ValueError(f'Missing pipeline name for pipeline number {i}')
            if not isinstance(p.steps[-1].step, assignments.AbstractAssignment):
                raise ValueError(f'Pipeline: {p.name}: the last step is not an instance '
                                 f'of \'assignments.AbstractAssignment\', got type: {type(p.steps[-1].step)}')

        base.check_unique_pipeline_names(pipelines=self.pipelines)

    def df_X_y(self, X, y) -> pd.DataFrame:
        return pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=self.columns)

    def get_repetitions(self) -> int:
        return self.repetitions

    def get_X_y_test(self, test_set_type: TestSetType) -> typing.Tuple[np.ndarray, np.ndarray]:
        if test_set_type == TestSetType.CLEAN_TEST_SET:
            return self.X_test, self.y_test
        else:
            return self.X_train_clean, self.y_train_clean

    def train_all_on_pipeline(self, estimator: TEstimator
                              ) -> CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep] | CleanPoisonedOutputPair[TrainSingleOutputEnsembleWithRep]:
        """
        Here we train an individual model (either a monolithic model or an ensemble with some pipeline)
        on all the clean and poisoned datasets we have.

        Training is performed with the given number of repetitions.
        :param estimator:
        :return:
        """

        pipeline_name = estimator.data_point_assignment.name if isinstance(
            estimator, models.EnsembleWithAssignmentPipeline) else (
                const.MONOLITHIC_ORACLED_PIPELINE_NAME)\
            if isinstance(estimator, models.EstimatorWithOracle) else const.MONOLITHIC_VANILLA_PIPELINE_NAME

        # first, train on the clean dataset.
        results_clean = self.train_model_with_rep(
            estimator=copy.deepcopy(estimator), X_train=self.X_train_clean, y_train=self.y_train_clean,
            poisoning_info=np.zeros_like(self.y_train_clean),
            info=base.ExpInfo(pipeline_name=pipeline_name,
                              perc_points=0.0, perc_features=0.0))

        with joblib.Parallel(n_jobs=len(self.poisoned_datasets)) as parallel:
            results_poisoned = parallel(joblib.delayed(self.train_model_with_rep)(
                estimator=copy.deepcopy(estimator),
                # this is pretty tricky, but basically
                # we are keeping the X part only.
                X_train=poisoned_dataset.sel(
                    y=[val for val in poisoned_dataset.coords['y'].values
                       if val not in const.DG_IRRELEVANT_COLUMNS]).to_numpy(),
                # no need to reshape this value.
                y_train=poisoned_dataset.sel(
                    y=base.const.COORD_LABEL
                ).to_numpy(),
                info=base.ExpInfo(
                    perc_points=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS],
                    perc_features=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES],
                    pipeline_name=pipeline_name
                ),
                poisoning_info=poisoned_dataset.sel(y=const.COORD_POISONED).to_numpy()
            ) for poisoned_dataset in self.poisoned_datasets.values())

        return CleanPoisonedOutputPair(clean=results_clean, poisoned=results_poisoned, pipeline_name=pipeline_name)


    def do(self) -> typing.Tuple[CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep],
    CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep],
    typing.List[CleanPoisonedOutputPair[TrainSingleOutputEnsembleWithRep]]]:
        """
        Top-level function training
        - the base (monolithic) model on all clean/poisoned datasets
        - risk-based ensemble according to the given pipeline
        - ground-truth risk-based ensemble according to the given pipeline
        - random-assignment ensemble
        :return:
        """

        # now we train the vanilla monolithic model.
        results_monolithic_vanilla: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep] = \
            self.train_all_on_pipeline(estimator=copy.deepcopy(self.monolithic_model))

        # now we train the oracle monolithic model.
        # the warning does not make any sense
        results_monolithic_oracle: CleanPoisonedOutputPair[TrainSingleOutputMonolithicWithRep] = \
            self.train_all_on_pipeline(estimator=models.EstimatorWithOracle(wrapped=copy.deepcopy(self.monolithic_model)))

        ensemble_pipelines = self.pipelines

        # now we train the ensemble on the required pipelines.
        with joblib.Parallel(n_jobs=len(ensemble_pipelines)) as parallel:
            results_ensemble = parallel(
                joblib.delayed(self.train_all_on_pipeline)(
                    estimator=models.EnsembleWithAssignmentPipeline(
                        base_estimator=sk_base.clone(self.monolithic_model), data_point_assignment=copy.deepcopy(p)))
                for p in self.pipelines)
        # just some unnecessary type-checking :)
        results_ensemble: typing.List[CleanPoisonedOutputPair[TrainSingleOutputEnsembleWithRep]] = \
            results_ensemble
        return results_monolithic_vanilla, results_monolithic_oracle, results_ensemble
