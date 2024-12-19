import copy
import typing

import pandas as pd
import pytest
from sklearn import ensemble

import models
import poisoning
from . import base, dataset_generator_test as dg_test, experiment_common as common,\
               experiment_common_test as common_test, experiment_ensemble_risk as exp
import pipe


def test_result_single_output_single_test_set_ensemble():
    N = 5
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = common_test.fake_pipeline_with_custom_quality_metrics(N=N)

    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')

    expected_risk_quality = base.risk_quality(risk_values=poisoning_info, y_true=poisoning_info, )

    expected_assignment_quality = common_test.standard_assignment_quality_non_agg(
        info=info, additional_metrics=custom_quality_metrics)

    expected_risk_quality = pd.concat([expected_risk_quality, expected_assignment_quality])

    expected = exp.TrainSingleOutputSingleTestSetEnsembleRisk(
        model_quality=common_test.standard_model_quality_non_agg(info=info),
        assignment_quality=expected_risk_quality, test_set_type=common.TestSetType.CLEAN_TEST_SET, info=info)

    fake_estimator = models.EnsembleWithAssignmentPipeline(
        base_estimator=ensemble.RandomForestClassifier(),
        data_point_assignment=fake_pipeline)
    fake_estimator.N_ = N

    got = exp.TrainSingleOutputSingleTestSetEnsembleRisk.from_results(
        y_pred=y_pred, y_test=y_test, info=info, poisoning_info=poisoning_info,
        hard_count=hard_count, estimator=fake_estimator, X_train=X, test_set_type=common.TestSetType.CLEAN_TEST_SET)

    common_test.evaluate_single_output_ensemble(expected=expected, got=got, info=info)


def test_train_single_output_ensemble_with_rep():
    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')
    N = 5
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = common_test.fake_pipeline_with_custom_quality_metrics(N=N)

    expected = common.TrainSingleOutputEnsembleWithRep(
        info=info,
        model_quality=common_test.model_quality_builder(info=info),
        assignment_quality=common_test.assignment_quality_builder(info=info,
                                                                  custom_quality_metrics=custom_quality_metrics))

    raw_results = [
        exp.TrainSingleOutputEnsembleRisk(
            info=info,
            results=[
                common_test.build_train_single_output_single_test_set_ensemble(
                    info=info, test_set_type=common.TestSetType.CLEAN_TEST_SET,
                    clazz=exp.TrainSingleOutputSingleTestSetEnsembleRisk, custom_quality_metrics=custom_quality_metrics),
                common_test.build_train_single_output_single_test_set_ensemble(
                    info=info, test_set_type=common.TestSetType.CLEAN_TRAINING_SET,
                    clazz=exp.TrainSingleOutputSingleTestSetEnsembleRisk, custom_quality_metrics=custom_quality_metrics),
            ]
        )
    for _ in range(2)]

    # ok, the error is fine.
    got = common.TrainSingleOutputEnsembleWithRep.from_reps(reps=raw_results,)
    common_test._evaluate_train_single_output_ensemble_with_rep(got=got, expected=expected, info=info)

def get_exp_from_dg(poisoning_generation_input: poisoning.PoisoningGenerationInput, base_model, pipes,
                    rep):
    ground_truth_pipelines = get_ground_truth_pipelines_from_pipes(pipes=pipes)
    dg = dg_test.get_dg(poisoning_generation_input=poisoning_generation_input)
    exp_ = exp.ExperimentEnsembleRisk.from_dataset_generator(dg=dg, monolithic_model=base_model,
                                                             pipelines=pipes, repetitions=rep,
                                                             ground_truth_pipelines=ground_truth_pipelines)
    return dg, exp_


def get_ground_truth_pipelines_from_pipes(pipes: typing.List[pipe.ExtPipeline]) -> typing.List[pipe.ExtPipeline]:
    ground_truth_pipelines = []
    for p in pipes:
        p_ground_truth = copy.deepcopy(p)
        p_ground_truth.full_name = f'GT({p.name})'
        ground_truth_pipelines.append(p_ground_truth)
    return ground_truth_pipelines


@pytest.mark.parametrize('estimator', [
    ensemble.RandomForestClassifier(),
    models.EnsembleWithAssignmentPipeline(base_estimator=ensemble.RandomForestClassifier(),
                                          data_point_assignment=common_test.simplest_pipeline())
])
def test_train_model_with_rep(estimator):
    get_dg_and_exp_kwargs = {
        'poisoning_generation_input': poisoning.PoisoningGenerationInput(
            perc_data_points=[10.0, 11.0],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty),
    }

    common_test._test_train_model_with_rep(estimator=estimator, get_dg_and_exp_kwargs=get_dg_and_exp_kwargs,
                                           gen_func=get_exp_from_dg)


@pytest.mark.parametrize('estimator', [
    ensemble.RandomForestClassifier(),
    models.EnsembleWithAssignmentPipeline(base_estimator=ensemble.RandomForestClassifier(),
                                          data_point_assignment=common_test.simplest_pipeline())
])
def test_train_all_on_pipeline(estimator):
    get_dg_and_exp_kwargs = {
        'poisoning_generation_input': poisoning.PoisoningGenerationInput(
            perc_data_points=[10.0, 10.5],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty),
    }

    common_test._test_train_all_on_pipeline(estimator=estimator, get_dg_and_exp_kwargs=get_dg_and_exp_kwargs,
                                            gen_func=get_exp_from_dg)


def test_do():
    pipes = [common_test.simplest_pipeline('p1'), common_test.simplest_pipeline('p2')]

    get_dg_and_exp_kwargs = {
        'poisoning_generation_input': poisoning.PoisoningGenerationInput(
            perc_data_points=[8.1, 8.2],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty),
    }

    # *2 because we create the same number of baselines for each pipe.
    common_test._test_do(pipes=pipes, expected_ensemble_len=len(pipes) * 2, gen_func=get_exp_from_dg,
                         exp_to_pipelines_func=lambda e: e.pipelines + e.ground_truth_pipelines,
                         get_dg_and_exp_kwargs=get_dg_and_exp_kwargs)
