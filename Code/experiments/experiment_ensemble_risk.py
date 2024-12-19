
import copy
import dataclasses
import enum
import typing
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn import base as sk_base
import xarray as xr

import experiments.experiment_common
import models
import pipe
import utils
from . import base, dataset_generator, experiment_common as common


class PipelineType(enum.Enum):
    NORMAL = 'NORMAL'
    GROUND_TRUTH = 'GROUND_TRUTH'

    def get_risk(self, retrieved_risk: np.ndarray, ground_truth_risk: np.ndarray):
        if self == PipelineType.GROUND_TRUTH:
            return ground_truth_risk
        return retrieved_risk


@dataclasses.dataclass
class TrainSingleOutputSingleTestSetEnsembleRisk: #(experiments.experiment_common.TrainSingleOutputEnsemble):
    """
    The output of an ensemble-based training (single execution) tested against an individual test set.

    Attributes:
        model_quality : pd.Series it includes the pipeline name percentage of poisoning
        assignment_quality : pd.Series it includes the pipeline name percentage of poisoning
        test_set_type : common.TestSetType the type of test set used for the evaluation.
        info : base.ExpInfo
    """
    # it includes the pipeline name percentage of poisoning
    model_quality: pd.Series
    # it includes the pipeline name percentage of poisoning
    assignment_quality: pd.Series
    # the type of test set used for the evaluation.
    test_set_type: common.TestSetType
    info: base.ExpInfo

    @staticmethod
    def from_results(*,
                     X_train,
                     estimator: models.EnsembleWithAssignmentPipeline,
                     test_set_type: common.TestSetType,
                     y_pred, y_test,
                     info: base.ExpInfo,  # X_y: pd.DataFrame,
                     hard_count: np.ndarray,
                     poisoning_info: np.ndarray) -> "TrainSingleOutputSingleTestSetEnsembleRisk":
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
        base_info = experiments.experiment_common.TrainSingleOutputSingleTestSetEnsemble.from_results(
            estimator=estimator, y_pred=y_pred, y_test=y_test, info=info, hard_count=hard_count,
            poisoning_info=poisoning_info, X_train=X_train, test_set_type=test_set_type)

        p, last_step = experiments.experiment_common.get_last_step_from_estimator(estimator=estimator)
        risk_quality_s, binarized_risk = base.extract_and_evaluate_risk(p=p, poisoning_info=poisoning_info)

        # now we have to concat assignment_recall and risk_quality to assignment_quality
        assignment_quality_s = last_step.get_custom_quality_metrics(X_train=X_train, y_pred=y_pred, y_test=y_test,
                                                                    poisoning_idx=poisoning_info,
                                                                    risk_values=binarized_risk,
                                                                    with_risk=True)
        if assignment_quality_s is None:
            # the output may be None. So, we create an empty one by default.
            # assignment_quality_s = pd.Series()
            assignment_quality_s = risk_quality_s
        else:
            assignment_quality_s = pd.concat([assignment_quality_s, risk_quality_s])
        assignment_quality_s = pd.concat([base_info.assignment_quality, assignment_quality_s])

        assert isinstance(assignment_quality_s, pd.Series)

        return TrainSingleOutputSingleTestSetEnsembleRisk(assignment_quality=assignment_quality_s, info=info,
                                             model_quality=base_info.model_quality, test_set_type=test_set_type)


@dataclasses.dataclass
class TrainSingleOutputEnsembleRisk:
    info: base.ExpInfo
    results: typing.List[TrainSingleOutputSingleTestSetEnsembleRisk]


TEstimator = typing.TypeVar('TEstimator', bound=utils.EstimatorProtocol)


@dataclasses.dataclass
class ExportConfigExpEnsembleRisk(base.AbstractExportConfig):
    pass


# this is necessary
@dataclasses.dataclass
class AnalyzedResultsEnsembleRisk(common.AnalyzedResultsEnsembleCommon):
    pass


class ExperimentEnsembleRisk(common.AbstractCommonExperiment, typing.Generic[TEstimator]):

    def __init__(self, repetitions: int,
                 monolithic_model: TEstimator,
                 X_train_clean: np.ndarray,
                 y_train_clean: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 poisoned_datasets: xr.Dataset,
                 pipelines: typing.List[pipe.ExtPipeline],
                 ground_truth_pipelines: typing.List[pipe.ExtPipeline],
                 columns: typing.Optional[typing.List[str]] = None):

        super().__init__(repetitions=repetitions, monolithic_model=monolithic_model,
                         X_train_clean=X_train_clean, y_train_clean=y_train_clean,
                         X_test=X_test, y_test=y_test, poisoned_datasets=poisoned_datasets,
                         pipelines=pipelines, columns=columns)
        self.ground_truth_pipelines = ground_truth_pipelines

        pipeline_names = {p.name for p in self.pipelines}
        ground_truth_pipeline_names = {p.name for p in self.ground_truth_pipelines}
        if len(pipeline_names.intersection(ground_truth_pipeline_names)) != 0:
            raise ValueError('Some names are in common between the pipelines and the baseline pipelines: '
                             f'{pipeline_names.intersection(ground_truth_pipeline_names)}')

    @property
    def analysis_class(self) -> typing.Type[AnalyzedResultsEnsembleRisk]:
        return AnalyzedResultsEnsembleRisk

    @staticmethod
    def from_dataset_generator(dg: dataset_generator.DatasetGenerator,
                               pipelines: typing.List[pipe.ExtPipeline],
                               ground_truth_pipelines: typing.List[pipe.ExtPipeline],
                               monolithic_model: TEstimator,
                               repetitions: int) -> "ExperimentEnsembleRisk":
        return ExperimentEnsembleRisk(X_test=dg.X_test, y_test=dg.y_test, X_train_clean=dg.X_train_clean,
                                      y_train_clean=dg.y_train_clean, monolithic_model=monolithic_model,
                                      pipelines=pipelines, ground_truth_pipelines=ground_truth_pipelines,
                                      poisoned_datasets=dg.all_datasets, columns=dg.columns, repetitions=repetitions)


    def _callback_ensemble(self, estimator, X_train, y_train, info: base.ExpInfo, poisoning_info: np.ndarray):
        # if it is an ensemble with pipeline, then we train it differently.
        if isinstance(estimator, models.EnsembleWithAssignmentPipelineGroundTruth) or isinstance(
                estimator, models.EnsembleWithAssignmentPipeline):
            # if it is a ground-truth ensemble, we need to set the risk ground truth that
            # corresponds to poisoned data points.
            if isinstance(estimator, models.EnsembleWithAssignmentPipelineGroundTruth):
                estimator.risk_ground_truth = poisoning_info


            estimator.fit(X=X_train, y=y_train)
                # raised = True

            # now we evaluate the model against the different test sets.
            results_ = []
            for (X_test, y_test, test_set_type) in [(self.X_test, self.y_test, common.TestSetType.CLEAN_TEST_SET),
                                                    (self.X_train_clean, self.y_train_clean,
                                                     common.TestSetType.CLEAN_TRAINING_SET)]:
                y_pred, count = estimator.hard_predictions_count(X_test)
                # if not raised:
                #     y_pred, count = estimator.hard_predictions_count(X_test)
                # else:
                # y_pred, count = np.repeat(np.nan, len(y_test)), np.repeat(np.nan, len(y_test))

                result = TrainSingleOutputSingleTestSetEnsembleRisk.from_results(
                    estimator=estimator, y_test=y_test, y_pred=y_pred, hard_count=count,
                    poisoning_info=poisoning_info, info=info, X_train=X_train, test_set_type=test_set_type)
                results_.append(result)
            return TrainSingleOutputEnsembleRisk(results=results_, info=info)


    def do(self
           ) -> typing.Tuple[
        experiments.experiment_common.CleanPoisonedOutputPair[
            experiments.experiment_common.TrainSingleOutputMonolithicWithRep],
        experiments.experiment_common.CleanPoisonedOutputPair[
            experiments.experiment_common.TrainSingleOutputMonolithicWithRep],
        typing.List[experiments.experiment_common.CleanPoisonedOutputPair[
            experiments.experiment_common.TrainSingleOutputEnsembleWithRep]]]:
        """
        Top-level function training
        - the base (monolithic) model on all clean/poisoned datasets
        - risk-based ensemble according to the given pipeline
        - ground-truth risk-based ensemble according to the given pipeline
        - random-assignment ensemble
        :return:
        """

        # now we train the monolithic and ensemble models.
        results_monolithic_vanilla, results_monolithic_oracle, results_ensemble = super().do()
        # just some unnecessary type-checking :)
        results_ensemble: typing.List[common.CleanPoisonedOutputPair[common.TrainSingleOutputEnsembleWithRep]] = \
            results_ensemble

        # here we train the ground-truth pipelines.
        # max(baselines, 1): so in case there are no baselines, it still works.
        with joblib.Parallel(n_jobs=max(len(self.ground_truth_pipelines), 1)) as parallel:
            results_ensemble_ground_truth = parallel(
                joblib.delayed(self.train_all_on_pipeline)(
                    estimator=models.EnsembleWithAssignmentPipelineGroundTruth(
                        base_estimator=sk_base.clone(self.monolithic_model), data_point_assignment=copy.deepcopy(p)))
                for p in self.ground_truth_pipelines)

        # just some unnecessary type-checking :)
        results_ensemble_ground_truth: typing.List[experiments.experiment_common.CleanPoisonedOutputPair[
            experiments.experiment_common.TrainSingleOutputEnsembleWithRep]] = \
            results_ensemble_ground_truth
        # results on "standard pipeline" and baseline are concat. They do not have special treatment
        return results_monolithic_vanilla, results_monolithic_oracle, results_ensemble + results_ensemble_ground_truth
