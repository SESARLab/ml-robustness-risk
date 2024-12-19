import copy
import os
import tempfile
import typing

import numpy as np
import pandas as pd
import pytest
from sklearn import ensemble

import assignments
import const
from . import base, dataset_generator_test as dg_test, experiment_common as common, \
    experiment_ensemble_plain_advanced as exp_plain, experiment_ensemble_risk as exp_risk
import models
import pipe
import poisoning


def standard_assignment_quality_agg(info: base.ExpInfo, additional_metrics: typing.Optional[pd.Series] = None) -> pd.Series:
    basic = standard_assignment_quality_non_agg(info, additional_metrics=additional_metrics)
    # very complicated, but we are just taking the first series, renaming columns adding AVG as prefix,
    # except when it refers to the pipeline name, perc_points, perc_features.
    # Then, we append to such as series the std(s), which are all zeros. We thus rename the columns adding STD as
    # prefix as well.
    result = pd.concat([basic.rename(lambda col: f'{const.PREFIX_AVG}({col})' if col not in const.INFO_KEYS_SET else col),
                        pd.Series(np.zeros(len(basic)-3), index=[f'{const.PREFIX_STD}({col})'
                                                                 for col in basic.index if col not in const.INFO_KEYS_SET])])
    return result

def standard_model_quality_agg(info: base.ExpInfo) -> pd.Series:
    """
    Utility function to generate a standard "model quality" with avg and std of the metrics.
    :param info:
    :return:
    """
    return info.prepend_to(pd.Series(
        np.concatenate([np.ones(len(base.METRICS_NAME)), np.zeros(len(base.METRICS_NAME))]),
        index=[f'{const.PREFIX_AVG}({col})' for col in base.METRICS_NAME] +
              [f'{const.PREFIX_STD}({col})' for col in base.METRICS_NAME]))

def standard_model_quality_non_agg(info: base.ExpInfo) -> pd.Series:
    return info.prepend_to(pd.Series(np.ones(len(base.METRICS_NAME)),
                                                index=base.METRICS_NAME))

def standard_assignment_quality_non_agg(info: base.ExpInfo, additional_metrics: typing.Optional[pd.Series] = None) -> pd.Series:
    concordance = pd.Series([1.0, 0.0, 0],
                                     index=[base.KEY_MAJORITY_COUNT_AVG,  base.KEY_MAJORITY_COUNT_STD,
                                            base.KEY_COUNT_N_DISCORDANT])

    # if additional_metrics is None, we create a "standard" version
    if additional_metrics is None:
        data = []
        for metric_name in assignments.CUSTOM_SCORE_ALL:
            data.append(0 if const.PREFIX_AVG in metric_name else 1)
        additional_metrics = pd.Series(data, index=assignments.CUSTOM_SCORE_ALL)

    return info.prepend_to(pd.concat([concordance, additional_metrics]))

def simplest_pipeline(name: str = 'p1') -> pipe.ExtPipeline:
    return pipe.ExtPipeline(steps=[pipe.Step('last', assignments.AssignmentRoundRobinBlind(N=3))], name=name,
                            pre_assignment_idx=0)


def check_and_compare_df(got: pd.DataFrame, expected: pd.DataFrame):
    got = got.reindex(sorted(got.columns), axis='columns')
    expected = expected.reindex(sorted(expected.columns), axis='columns')
    # somehow the pd.DataFrame.equals does not help.
    # result = got.equals(expected)
    result = np.all(got == expected)
    if not result:
        print(f'Diff\n{got.compare(expected)}')
    assert result


def test_result_train_single_output_single_test_set_monolithic():
    """
    Very simple test checking the creation of TrainSingleOutputSingleTestSetMonolithic.
    This is an execution and evaluation on an individual test set.
    """
    y_pred = np.ones(10)
    y_test = np.ones(10)

    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')

    expected = common.TrainSingleOutputSingleTestSetMonolithic(
        model_quality=standard_model_quality_non_agg(info), test_set_type=common.TestSetType.CLEAN_TEST_SET
    )

    got = common.TrainSingleOutputSingleTestSetMonolithic.from_results(y_pred=y_pred, y_test=y_test, info=info,
                                                                       test_set_type=common.TestSetType.CLEAN_TEST_SET)
    assert np.all(expected.model_quality == got.model_quality)
    assert np.all(expected.model_quality.index == got.model_quality.index)
    assert np.all(expected.test_set_type == got.test_set_type)


def evaluate_single_output_ensemble(
        expected: common.TrainSingleOutputSingleTestSetEnsemble | exp_risk.TrainSingleOutputSingleTestSetEnsembleRisk,
        got: common.TrainSingleOutputSingleTestSetEnsemble | exp_risk.TrainSingleOutputSingleTestSetEnsembleRisk,
        info: base.ExpInfo,):

    # we sort the indices of model and assignment quality just to be sure
    for target in [expected.assignment_quality, expected.model_quality, got.assignment_quality, got.model_quality]:
        target.sort_index(inplace=True)

    assert got.info == info
    assert expected.model_quality.equals(got.model_quality)
    assert expected.test_set_type == got.test_set_type
    assert expected.assignment_quality.equals(got.assignment_quality)


def fake_pipeline_with_custom_quality_metrics(N: int, X: typing.Optional[np.ndarray] = None,
                                              y_pred: typing.Optional[np.ndarray] = None,
                                              y_test: typing.Optional[np.ndarray] = None,
                                              poisoning_idx: typing.Optional[np.ndarray] = None,
                                              risk_values: typing.Optional[np.ndarray] = None, ):
    if y_pred is None:
        y_pred = np.ones(10)
    if y_test is None:
        y_test = np.ones(10)
    if X is None:
        X = np.random.default_rng().random(size=(10, 5))
    if poisoning_idx is None:
        poisoning_idx = np.random.default_rng().choice([0, 1], size=(len(X),))
    # if risk_values is None:
    #     risk_values = poisoning_idx

    fake_pipeline = pipe.ExtPipeline(
        name='p1',
        # pre_assignment_idx=pre_assignment_idx,
        steps=[
            pipe.Step('s1', assignments.AssignmentRoundRobinBlind(N=N))
        ])

    # specific values for the assignment to obtain specific assignment values (just use tile and that's it!)
    assignment = np.tile(np.arange(N), int(len(y_test) / N))
    last_step: assignments.AbstractAssignment = fake_pipeline.steps[-1].step
    last_step.assignment_ = assignment
    last_step.assigned_to_each_ = np.bincount(assignment)
    custom_quality_metrics = last_step.get_custom_quality_metrics(X_train=X, y_test=y_test, y_pred=y_pred,
                                                                  poisoning_idx=poisoning_idx, risk_values=None,
                                                                  with_risk=False)
    hard_count = np.ones(10)
    return X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, poisoning_idx


def test_result_train_single_output_single_test_set_ensemble():
    """
    Very simple test checking the creation of TrainSingleOutputSingleTestSetEnsemble.
    This is an execution and evaluation on an individual test set.
    """
    N = 5
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = fake_pipeline_with_custom_quality_metrics(N=N)

    # hard_count = np.ones(10)

    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')

    # values of the diversity must be retrieved as above because the alternative is building
    # the assignment in a way that... but we are not interested in such goal here, so we just
    # rely on function standard_assignment_quality_non_agg
    expected = common.TrainSingleOutputSingleTestSetEnsemble(
        info=info, test_set_type=common.TestSetType.CLEAN_TEST_SET,
        model_quality=standard_model_quality_non_agg(info),
        assignment_quality=standard_assignment_quality_non_agg(info=info, additional_metrics=custom_quality_metrics),
    )

    fake_estimator = models.EnsembleWithAssignmentPipeline(
        base_estimator=ensemble.RandomForestClassifier(),
        data_point_assignment=fake_pipeline)
    fake_estimator.N_ = N

    got = common.TrainSingleOutputSingleTestSetEnsemble.from_results(
        y_pred=y_pred, y_test=y_test, info=info, poisoning_info=poisoning_info,
        hard_count=hard_count, test_set_type=common.TestSetType.CLEAN_TEST_SET, estimator=fake_estimator, X_train=X)

    evaluate_single_output_ensemble(expected=expected, got=got, info=info)


def test_result_single_output_monolithic_with_rep():
    """
    Simple test checking the creation of TrainSingleOutputMonolithicWithRep.
    This is a set of executions and evaluations on different test sets.
    """
    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')

    expected = common.TrainSingleOutputMonolithicWithRep(
        info=info,
        model_quality={
            common.TestSetType.CLEAN_TEST_SET: standard_model_quality_agg(info=info),
            common.TestSetType.CLEAN_TRAINING_SET: standard_model_quality_agg(info=info),
        })

    raw_results = [common.TrainSingleOutputMonolithic(
        info=info,
        results=[
            # we must have at least two occurrences otherwise ths std is nan and the assertion fails.
            common.TrainSingleOutputSingleTestSetMonolithic(
                test_set_type=common.TestSetType.CLEAN_TEST_SET, model_quality=standard_model_quality_non_agg(info)),
            common.TrainSingleOutputSingleTestSetMonolithic(
                test_set_type=common.TestSetType.CLEAN_TRAINING_SET, model_quality=standard_model_quality_non_agg(info)),
        ]) for _ in range(2)]

    got = common.TrainSingleOutputMonolithicWithRep.from_reps(reps=raw_results)

    assert got.info == info
    for test_set_type, model_quality in expected.model_quality.items():
        assert expected.model_quality[test_set_type].equals(got.model_quality[test_set_type])


def _evaluate_train_single_output_ensemble_with_rep(
        got: common.TrainSingleOutputEnsembleWithRep,
        expected: common.TrainSingleOutputEnsembleWithRep, info: base.ExpInfo):
    assert got.info == info
    for test_set_type in expected.model_quality.keys():
        assert expected.model_quality[test_set_type].sort_index(inplace=False).equals(
            got.model_quality[test_set_type].sort_index(inplace=False))
        assert expected.assignment_quality[test_set_type].sort_index(inplace=False).equals(
            got.assignment_quality[test_set_type].sort_index(inplace=False))


def build_train_single_output_single_test_set_ensemble(
        info: base.ExpInfo, custom_quality_metrics: pd.Series, test_set_type: common.TestSetType,
        clazz: typing.Type[common.TrainSingleOutputSingleTestSetEnsemble | exp_risk.TrainSingleOutputSingleTestSetEnsembleRisk]):
    return clazz(
        test_set_type=test_set_type, model_quality=standard_model_quality_non_agg(info), info=info,
        assignment_quality=standard_assignment_quality_non_agg(info, additional_metrics=custom_quality_metrics))



def test_result_train_single_output_ensemble_with_rep():
    """
    Simple test checking the creation of TrainSingleOutputEnsembleWithRep.
    """
    info = base.ExpInfo(perc_features=10, perc_points=10, pipeline_name='p1')
    N = 5
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = fake_pipeline_with_custom_quality_metrics(N=N)

    # what we expect.
    expected = common.TrainSingleOutputEnsembleWithRep(
        info=info,
        model_quality=model_quality_builder(info=info),
        assignment_quality=assignment_quality_builder(info=info, custom_quality_metrics=custom_quality_metrics),
    )

    # the result results we pass as input.
    raw_results = [common.TrainSingleOutputEnsemble(
        info=info,
        results=[
            build_train_single_output_single_test_set_ensemble(info=info, test_set_type=common.TestSetType.CLEAN_TEST_SET,
                                                               clazz=common.TrainSingleOutputSingleTestSetEnsemble,
                                                               custom_quality_metrics=custom_quality_metrics),
            build_train_single_output_single_test_set_ensemble(info=info,
                                                               test_set_type=common.TestSetType.CLEAN_TRAINING_SET,
                                                               clazz=common.TrainSingleOutputSingleTestSetEnsemble,
                                                               custom_quality_metrics=custom_quality_metrics)
            ]
    ) for _ in range(2)]

    got = common.TrainSingleOutputEnsembleWithRep.from_reps(reps=raw_results)
    _evaluate_train_single_output_ensemble_with_rep(got=got, expected=expected, info=info)



def _test_pair_result_advanced(
        pair: common.AnalyzedOutputPairMonolithic | common.AnalyzedOutputPairEnsemble,
        info_clean: base.ExpInfo,
        info_poisoned: typing.List[base.ExpInfo]
):
    """
    we receive as input the info because they are the only thing that are important here.
    :param pair:
    :param info_clean:
    :param info_poisoned:
    :return:
    """
    for test_set_type in pair.model_quality.keys():
        # y is INFO_KEYS_SET (pipeline, perc_points, perc_features) + 2 * (metrics).
        # The reason of the multiplication is: we have both avg and std for each metric.
        assert pair.model_quality[test_set_type].shape == (len(info_poisoned) + 1, len(const.INFO_KEYS_SET) + len(base.METRICS) * 2)
        # same as above, but here we don't have std.
        assert pair.delta_self[test_set_type].shape == (len(info_poisoned), len(const.INFO_KEYS_SET) + len(base.METRICS))

        # very picky checks...
        assert np.all(pair.delta_self[test_set_type][const.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
        assert np.all(pair.delta_self[test_set_type][const.KEY_PERC_DATA_POINTS] == [info.perc_points for info in info_poisoned])
        assert np.all(pair.delta_self[test_set_type][const.KEY_PERC_FEATURES] == [info.perc_features for info in info_poisoned])

        source = [pair.model_quality[test_set_type]]

        # now, if we are receiving results referred to the monolithic with oracle/ensemble,
        # there are more deltas to check
        if isinstance(pair, common.AnalyzedOutputPairMonolithicWithOracle):

            assert pair.delta_ref_oracled[test_set_type].shape == (
                len(info_poisoned) + 1, len(const.INFO_KEYS_SET) + len(base.METRICS))
            # y is INFO_KEYS_SET (pipeline, perc_points, perc_features) + 2 * (metrics).
            # The reason of the multiplication is: we have both avg and std for each metric.
            assert pair.delta_self_oracled[test_set_type].shape == (
                len(info_poisoned) + 1, len(const.INFO_KEYS_SET) + len(base.METRICS) * 2)

        elif isinstance(pair, common.AnalyzedOutputPairEnsemble):
            # we have the difference with respect to the monolithic model
            # across all percentages of poisoning (len(result_poisoned)) and clean as well (+1) -> first dimension.
            # In the second dimension, we have INFO_KEYS_SET + the metrics on which we retrieve the difference.
            print(pair.delta_ref_monolithic_vanilla[test_set_type].columns)
            assert pair.delta_ref_monolithic_vanilla[test_set_type].shape == (
                len(info_poisoned) + 1, len(const.INFO_KEYS_SET) + len(base.METRICS))
            assert pair.delta_ref_monolithic_oracled[test_set_type].shape == (
                len(info_poisoned) + 1, len(const.INFO_KEYS_SET) + len(base.METRICS))

            source.append(pair.delta_ref_monolithic_vanilla[test_set_type])
            source.append(pair.delta_ref_monolithic_oracled[test_set_type])

        # now we check also for model quality and, if present, for delta_monolithic.
        # Note: these results are available also for the clean dataset, so we have one row more
        # compared to delta_self.
        for single_source in source:
            assert np.all(single_source[const.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
            assert np.all(single_source[const.KEY_PERC_DATA_POINTS] == [info_clean.perc_points] +
                          [info.perc_points for info in info_poisoned])
            assert np.all(single_source[const.KEY_PERC_DATA_POINTS] == [info_clean.perc_features] +
                          [info.perc_features for info in info_poisoned])
#
#
def test_pair_monolithic():
    N = 5
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = fake_pipeline_with_custom_quality_metrics(N=N)

    info_clean = base.ExpInfo(perc_features=0, perc_points=0, pipeline_name=fake_pipeline.name)
    info_poisoned = [base.ExpInfo(perc_features=10, perc_points=10, pipeline_name=fake_pipeline.name),
                     base.ExpInfo(perc_features=15, perc_points=15, pipeline_name=fake_pipeline.name),
                     base.ExpInfo(perc_features=20, perc_points=20, pipeline_name=fake_pipeline.name)]

    clean = common.TrainSingleOutputMonolithicWithRep(
        model_quality=model_quality_builder(info=info_clean), info=info_clean)

    poisoned = [common.TrainSingleOutputMonolithicWithRep(
        model_quality=model_quality_builder(info=info), info=info) for info in info_poisoned
    ]

    pair = common.AnalyzedOutputPairMonolithic.from_results(
        result=common.CleanPoisonedOutputPair(clean=clean,
                                              poisoned=poisoned,
                                              pipeline_name=fake_pipeline.name))

    _test_pair_result_advanced(pair=pair, info_poisoned=info_poisoned, info_clean=info_clean)

def model_quality_builder(info: base.ExpInfo):
    # just to simplify the code
    return {
            common.TestSetType.CLEAN_TEST_SET: standard_model_quality_agg(info=info),
            common.TestSetType.CLEAN_TRAINING_SET: standard_model_quality_agg(info=info)
        }

def assignment_quality_builder(info: base.ExpInfo, custom_quality_metrics: typing.Optional[pd.Series] = None):
    return {
        common.TestSetType.CLEAN_TEST_SET: standard_assignment_quality_agg(
            info=info, additional_metrics=custom_quality_metrics),
        common.TestSetType.CLEAN_TRAINING_SET: standard_assignment_quality_agg(
            info=info, additional_metrics=custom_quality_metrics)
    }

def test_pair_ensemble():
    X, y_test, y_pred, fake_pipeline, custom_quality_metrics, hard_count, \
        poisoning_info = fake_pipeline_with_custom_quality_metrics(N=5)

    info_clean = base.ExpInfo(perc_features=0, perc_points=0, pipeline_name=fake_pipeline.name)
    info_poisoned = [base.ExpInfo(perc_features=10, perc_points=10, pipeline_name=fake_pipeline.name),
                     base.ExpInfo(perc_features=15, perc_points=15, pipeline_name=fake_pipeline.name),
                     base.ExpInfo(perc_features=20, perc_points=20, pipeline_name=fake_pipeline.name),
                     base.ExpInfo(perc_features=25, perc_points=25, pipeline_name=fake_pipeline.name),
                     ]

    clean = common.TrainSingleOutputEnsembleWithRep(
        model_quality=model_quality_builder(info=info_clean),
        assignment_quality=assignment_quality_builder(info=info_clean, custom_quality_metrics=custom_quality_metrics),
        info=info_clean)

    poisoned = [
        common.TrainSingleOutputEnsembleWithRep(
            model_quality=model_quality_builder(info=info),
            assignment_quality=assignment_quality_builder(
                info=info, custom_quality_metrics=custom_quality_metrics), info=info)
        for info in info_poisoned]

    # to test the creation of "the ensemble pair result", we also need results
    # from the base model. Note: it is necessary to retrieve the "new delta" (i.e., referred to the base model).
    pair_monolithic_vanilla = common.CleanPoisonedOutputPair(clean=clean, poisoned=poisoned, pipeline_name=fake_pipeline.name)
    pair_monolithic_oracled = common.CleanPoisonedOutputPair(clean=clean, poisoned=poisoned, pipeline_name=fake_pipeline.name)

    pair = common.AnalyzedOutputPairEnsemble.from_results(
        result=common.CleanPoisonedOutputPair(clean=clean, poisoned=poisoned, pipeline_name=fake_pipeline.name),
        reference_monolithic_model_vanilla=pair_monolithic_vanilla, reference_monolithic_model_oracled=pair_monolithic_oracled,)

    _test_pair_result_advanced(pair=pair, info_poisoned=info_poisoned, info_clean=info_clean)


def check_and_compare_list(dir_name: str, got: typing.Iterable[str], expected: typing.Iterable[str]):
    result = sorted(got) == sorted(expected)
    if not result:
        print(f'[{dir_name}]: In expected but not in result:\n{set(expected) - set(got)}\n'
              f'In results but unexpected:\n{set(got) - set(expected)}')
    assert result

#
@pytest.mark.parametrize('export_config', [
    common.ExportConfigExpEnsembleCommon(
        exists_ok=True,
    )
])
def test_analyzed_results(export_config: common.ExportConfigExpEnsembleCommon):
    # these info are used both for the plain model.
    info_monolithic_clean = base.ExpInfo(perc_features=0.0, perc_points=0.0, pipeline_name='plain')

    info_monolithic_poisoned = [base.ExpInfo(perc_features=0.0, perc_points=10, pipeline_name='plain'),
                          base.ExpInfo(perc_features=0.0, perc_points=15, pipeline_name='plain'),
                          base.ExpInfo(perc_features=0.0, perc_points=20, pipeline_name='plain')]

    info_ensemble_clean_1 = base.ExpInfo(perc_features=0.0, perc_points=0.0, pipeline_name='p1')
    info_ensemble_poisoned_1 = [base.ExpInfo(perc_features=0.0, perc_points=10, pipeline_name='p1'),
                                base.ExpInfo(perc_features=0.0, perc_points=15, pipeline_name='p1'),
                                base.ExpInfo(perc_features=0.0, perc_points=20, pipeline_name='p1')]

    info_ensemble_clean_2 = base.ExpInfo(perc_features=0.0, perc_points=0.0, pipeline_name='p2')
    info_ensemble_poisoned_2 = [base.ExpInfo(perc_features=0.0, perc_points=10, pipeline_name='p2'),
                                base.ExpInfo(perc_features=0.0, perc_points=15, pipeline_name='p2'),
                                base.ExpInfo(perc_features=0.0, perc_points=20, pipeline_name='p2')]

    info_ensemble_clean_3 = base.ExpInfo(perc_features=0.0, perc_points=0.0, pipeline_name='p3')
    info_ensemble_poisoned_3 = [base.ExpInfo(perc_features=0.0, perc_points=10, pipeline_name='p3'),
                                base.ExpInfo(perc_features=0.0, perc_points=15, pipeline_name='p3'),
                                base.ExpInfo(perc_features=0.0, perc_points=20, pipeline_name='p3')]

    pipeline_names_ensemble = ['p1', 'p2', 'p3']

    pair_monolithic_vanilla = common.CleanPoisonedOutputPair(
        pipeline_name=info_monolithic_clean.pipeline_name,
        clean=common.TrainSingleOutputMonolithicWithRep(info=info_monolithic_clean,
                                                        model_quality=model_quality_builder(info=info_monolithic_clean)),
        poisoned=[common.TrainSingleOutputMonolithicWithRep(
            model_quality=model_quality_builder(info=info), info=info) for info in info_monolithic_poisoned
        ])

    pair_monolithic_oracled = copy.deepcopy(pair_monolithic_vanilla)

    pair_ensemble = [
        common.CleanPoisonedOutputPair(
            pipeline_name=pipeline_names_ensemble[0],
            clean=common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info_ensemble_clean_1), info=info_ensemble_clean_1,
                assignment_quality=assignment_quality_builder(info=info_ensemble_clean_1)
            ),
            poisoned=[common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info), info=info,
                assignment_quality=assignment_quality_builder(info=info),
            ) for info in info_ensemble_poisoned_1]
        ),
        common.CleanPoisonedOutputPair(
            pipeline_name=pipeline_names_ensemble[1],
            clean=common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info_ensemble_clean_2), info=info_ensemble_clean_2,
                assignment_quality=assignment_quality_builder(info=info_ensemble_clean_2),
            ),
            poisoned=[common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info), info=info,
                assignment_quality=assignment_quality_builder(info=info),
            ) for info in info_ensemble_poisoned_2]
        ),
        common.CleanPoisonedOutputPair(
            pipeline_name=pipeline_names_ensemble[2],
            clean=common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info_ensemble_clean_3), info=info_ensemble_clean_3,
                assignment_quality=assignment_quality_builder(info=info_ensemble_clean_3),
            ),
            poisoned=[common.TrainSingleOutputEnsembleWithRep(
                model_quality=model_quality_builder(info=info), info=info,
                assignment_quality=assignment_quality_builder(info=info),
            ) for info in info_ensemble_poisoned_3]
        ),
    ]

    analyzed_results = common.AnalyzedResultsEnsembleCommon.from_results(
        results_monolithic_vanilla=pair_monolithic_vanilla, results_ensemble=pair_ensemble,
        results_monolithic_oracled=pair_monolithic_oracled,)

    pipeline_names_ensemble = set(pipeline_names_ensemble)

    # check the correctness of the pipeline names.
    sources = [analyzed_results.ensemble_model_quality, analyzed_results.ensemble_delta_self,
               analyzed_results.ensemble_delta_self, analyzed_results.assignment_quality]

    # from here things become interesting
    for source in sources:
        assert set(source.keys()) == pipeline_names_ensemble

    with tempfile.TemporaryDirectory() as temp_dir:
        export_config.base_directory = temp_dir

        analyzed_results.export(config=export_config)

        expected_base_dirs = [base.EXP_DIR_MERGED, base.EXP_DIR_DELTA_REFERENCE, base.EXP_DIR_DELTA_SELF,
                              base.EXP_DIR_MODEL_QUALITY, base.EXP_DIR_ASSIGNMENTS]

        for single_dir in expected_base_dirs:
            assert os.path.exists(os.path.join(temp_dir, single_dir)) and \
                   os.path.isdir(os.path.join(temp_dir, single_dir))

        # there should be the following files under the directory EXP_DIR_MERGED:
        # - ensemble_quality_training_set_clean.csv
        # - delta_ensemble_self_training_set_clean.csv
        # - ensemble_quality_test_set_clean.csv
        # - assignment_test_set_clean.csv
        # - assignment_training_set_clean.csv\
        # - ensemble_delta_ref_mono_vanilla_training_set_clean.csv
        # - ensemble_delta_ref_mono_vanilla_test_set_clean.csv
        # - ensemble_delta_ref_mono_oracle_training_set_clean.csv
        # - delta_ensemble_self_test_set_clean.csv
        # - ensemble_delta_ref_mono_oracle_test_set_clean.csv
        merged = os.listdir(os.path.join(temp_dir, base.EXP_DIR_MERGED))
        expected_file_list =[
            # delta self
            base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_SELF,
            base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF,
            base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF,
            # delta ref
            base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_VANILLA,
            base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_ORACLED,
            base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED,
            # assignment quality
            base.FILE_NAME_EXPORT_ENSEMBLE_ASSIGNMENT,
            base.FILE_NAME_EXPORT_ENSEMBLE_QUALITY
            ]
        expected_file_list = [f'{f}_{t.prefix()}.csv' for f in expected_file_list for t in common.TestSetType]
        check_and_compare_list(got=merged, expected=expected_file_list, dir_name=base.EXP_DIR_MERGED)

        # there should be x files under EXP_DIR_DELTA_REFERENCE, EXP_DIR_DELTA_SELF, EXP_DIR_ASSIGNMENTS,
        # and EXP_DIR_MODEL_QUALITY
        # In delta ref, we have:
        # - ensemble_delta_ref_mono_oracle_{pipeline_name}
        # - ensemble_delta_ref_vanilla_oracle_{pipeline_name}
        # - mono_vanilla_delta_ref_mono_oracle
        # In delta self, we have:
        # - delta_self_ensemble_{pipeline_name}
        # - mono_oracled_delta_self
        # - mono_vanilla_delta_self
        # In model quality, we have:
        # - ensemble_quality_{pipeline_name}
        # - mono_oracled_quality
        # - mono_vanilla_quality
        # In assignment, we have:
        # - assignment_{pipeline_name}
        # Each of the above is repeated for each test set type.
        for dir_to_check, expected_file_list in [
            (base.EXP_DIR_DELTA_REFERENCE,
                [f'{base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED}_{t.prefix()}.csv' for t in common.TestSetType]+
                [f'{f}_{p}_{t.prefix()}.csv'
                    for f in [base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_VANILLA, base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_REF_AGAINST_MONO_ORACLED]
                    for p in pipeline_names_ensemble
                    for t in common.TestSetType]),
            (base.EXP_DIR_DELTA_SELF,
                [f'{f}_{t.prefix()}.csv'
                    for f in [base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF]
                    for t in common.TestSetType] +
                [f'{base.FILE_NAME_EXPORT_ENSEMBLE_DELTA_SELF}_{p}_{t.prefix()}.csv'
                    for p in pipeline_names_ensemble for t in common.TestSetType]),
            (base.EXP_DIR_ASSIGNMENTS, [f'{base.FILE_NAME_EXPORT_ENSEMBLE_ASSIGNMENT}_{p}_{t.prefix()}.csv'
                                        for p in pipeline_names_ensemble
                                        for t in common.TestSetType]),
        ]:
            check_and_compare_list(got=os.listdir(os.path.join(temp_dir, dir_to_check)), expected=expected_file_list,
                                   dir_name=dir_to_check)


def get_exp_from_dg(poisoning_generation_input: poisoning.PoisoningGenerationInput, base_model, pipes,
                    rep):
    dg = dg_test.get_dg(poisoning_generation_input=poisoning_generation_input)
    exp = exp_plain.ExperimentEnsemblePlainAdvanced.from_dataset_generator(dg=dg, monolithic_model=base_model,
                                                                           pipelines=pipes, repetitions=rep)
    return dg, exp


def _test_train_model_with_rep(estimator, get_dg_and_exp_kwargs, gen_func=get_exp_from_dg):
    pipes_to_use = [
        estimator.data_point_assignment if isinstance(estimator, models.EnsembleWithAssignmentPipeline)
        else simplest_pipeline()]

    dg, exp = gen_func(base_model=estimator, pipes=pipes_to_use, rep=1, **get_dg_and_exp_kwargs)

    info = base.ExpInfo(pipeline_name='p1', perc_points=0.0, perc_features=0.0)

    poisoning_info = np.random.default_rng().choice([0, 1], size=len(dg.X_train_clean))

    result = exp.train_model_with_rep(estimator=estimator, X_train=dg.X_train_clean, y_train=dg.y_train_clean,
                                      info=info,  poisoning_info=poisoning_info)

    if isinstance(estimator, models.EnsembleWithAssignmentPipeline):
        assert isinstance(result, common.TrainSingleOutputEnsembleWithRep)
    else:
        assert isinstance(result, common.TrainSingleOutputMonolithicWithRep)
        # we can't do the check below because the estimator is cloned before training, and
        # we don't have access to such a cloned variable
        # if isinstance(estimator, models.EstimatorWithOracle):
        #     print('check inner')
        #     assert np.all(estimator.poisoning_info == poisoning_info)


@pytest.mark.parametrize('estimator', [
    ensemble.RandomForestClassifier(),
    models.EnsembleWithAssignmentPipeline(base_estimator=ensemble.RandomForestClassifier(),
                                          data_point_assignment=simplest_pipeline()),
    models.EstimatorWithOracle(wrapped=ensemble.RandomForestClassifier()),
])
def test_train_model_with_rep(estimator):

    get_dg_and_exp_kwargs = {
        'poisoning_generation_input': poisoning.PoisoningGenerationInput(
            perc_data_points=[10.0, 15.0, 20.0, 21.0],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty),
    }

    _test_train_model_with_rep(estimator=estimator, get_dg_and_exp_kwargs=get_dg_and_exp_kwargs)


def _test_train_all_on_pipeline(estimator, get_dg_and_exp_kwargs, gen_func=get_exp_from_dg):
    pipes_to_use = [
        estimator.data_point_assignment if isinstance(estimator, models.EnsembleWithAssignmentPipeline)
        else simplest_pipeline()]

    dg, exp = gen_func(base_model=estimator, pipes=pipes_to_use, rep=1, **get_dg_and_exp_kwargs)

    result = exp.train_all_on_pipeline(estimator=estimator)

    if isinstance(estimator, models.EnsembleWithAssignmentPipeline):
        assert isinstance(result.clean, common.TrainSingleOutputEnsembleWithRep)
    else:
        assert isinstance(result.clean, common.TrainSingleOutputMonolithicWithRep)


@pytest.mark.parametrize('estimator', [
    ensemble.RandomForestClassifier(),
    models.EnsembleWithAssignmentPipeline(base_estimator=ensemble.RandomForestClassifier(),
                                          data_point_assignment=simplest_pipeline()),
    models.EstimatorWithOracle(wrapped=ensemble.RandomForestClassifier()),
])
def test_train_all_on_pipeline(estimator):
    get_dg_and_exp_kwargs = {
        'poisoning_generation_input': poisoning.PoisoningGenerationInput(
            perc_data_points=[10.0, 15.0, 20.0, 21.0, 22.0],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty),
    }

    _test_train_all_on_pipeline(estimator=estimator, get_dg_and_exp_kwargs=get_dg_and_exp_kwargs)


def _test_do(pipes, get_dg_and_exp_kwargs, expected_ensemble_len,
             exp_to_pipelines_func: typing.Callable[[common.AbstractCommonExperiment], typing.List[pipe.ExtPipeline]],
             # callable that given the experiment returns the pipelines to consider
             gen_func=get_exp_from_dg,):

    dg, exp = gen_func(base_model=ensemble.RandomForestClassifier(), pipes=pipes, rep=1, **get_dg_and_exp_kwargs)

    result_monolithic_vanilla, result_monolithic_oracle, result_ensemble = exp.do()

    assert result_monolithic_vanilla.pipeline_name == const.MONOLITHIC_VANILLA_PIPELINE_NAME
    assert result_monolithic_oracle.pipeline_name == const.MONOLITHIC_ORACLED_PIPELINE_NAME

    # one result for each pipeline (containing all the variations over the poisoned datasets)
    assert len(result_ensemble) == expected_ensemble_len
    # one result for each poisoned dataset
    assert len(result_monolithic_vanilla.poisoned) == len(dg)
    assert len(result_monolithic_oracle.poisoned) == len(dg)

    for single_res_ensemble, single_pipe in zip(result_ensemble, exp_to_pipelines_func(exp)):
        assert single_res_ensemble.pipeline_name == single_pipe.name
        # one result for each poisoned dataset
        assert len(single_res_ensemble.poisoned) == len(dg)


def test_do():
    pipes = [simplest_pipeline('p1'), simplest_pipeline('p2')]

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

    _test_do(pipes=pipes, expected_ensemble_len=len(pipes), exp_to_pipelines_func=lambda e: e.pipelines,
             get_dg_and_exp_kwargs=get_dg_and_exp_kwargs)
