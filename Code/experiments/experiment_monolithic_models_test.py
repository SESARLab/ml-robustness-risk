import os
import tempfile
import typing

import numpy as np
import pandas as pd
from sklearn import ensemble, tree

import const
import poisoning
from . import base, dataset_generator_test as dg_test, experiment_common as common, experiment_common_test as common_test, \
    experiment_monolithic_models as exp


def build_quality_metrics(info: base.ExpInfo, multiplier: typing.Optional[float] = None) -> pd.Series:
    if multiplier is None:
        multiplier = 1
    return info.prepend_to(pd.Series(
                    np.concatenate([np.ones(len(base.METRICS_NAME)) * multiplier, np.zeros(len(base.METRICS_NAME))]),
                    index=[f'{const.PREFIX_AVG}({col})' for col in base.METRICS_NAME] + [
                        f'{const.PREFIX_STD}({col})' for col in base.METRICS_NAME]
                ))

HardcodedResultsType: typing.TypeAlias = typing.Dict[str, typing.Dict[str, typing.Dict[str,
    # typing.Union[typing.Dict[str, base.ExpInfo], typing.Dict[common.TestSetType, pd.DataFrame]]
    typing.Union[base.ExpInfo, typing.Dict[common.TestSetType, pd.DataFrame]]
]]]

def col_to_drop_for_delta(hardcoded_results: HardcodedResultsType, model_name_: str, test_set_type_: common.TestSetType):
        # these columns are not used in delta.
        return [col for col in hardcoded_results[model_name_]['0']['model_quality'][test_set_type_].index
                   if const.PREFIX_STD in col] + list(const.INFO_KEYS_SET)

def build_info_column_for_df(hardcoded_results: HardcodedResultsType,
                             df_: pd.DataFrame, model_name_: str, test_set_type_: common.TestSetType):
        # this function can be called both for delta_self and delta_ref. When we use delta_ref,
        # we have an additional column upfront, which is the one with eps=0.
        # These columns are always there.
        rows = [hardcoded_results[model_name_]['1'], hardcoded_results[model_name_]['2'],
                hardcoded_results[model_name_]['3']]
        if len(df_) == len(rows) + 1:
            rows = [hardcoded_results[model_name_]['0']] + rows

        df_[const.KEY_PERC_DATA_POINTS] = [row['model_quality'][test_set_type_][const.KEY_PERC_DATA_POINTS]
                                         for row in rows]
        df_[const.KEY_PERC_FEATURES] = [row['model_quality'][test_set_type_][const.KEY_PERC_FEATURES]
                                      for row in rows]
        return df_

def build_df_from_delta_diff(hardcoded_results: HardcodedResultsType,
                             diffs_, model_name_: str, test_set_type_: common.TestSetType):
        df_ = pd.DataFrame(diffs_)
        df_ = df_.rename(lambda col: f'{model_name_}_{const.PREFIX_DELTA}({col})', axis='columns')
        df_ = build_info_column_for_df(df_=df_, model_name_=model_name_, test_set_type_=test_set_type_,
                                       hardcoded_results=hardcoded_results)
        return df_

def build_expected_df_model_quality(hardcoded_results: HardcodedResultsType,
                                    model_name_, test_set_type_: common.TestSetType):
    # the first row is repeated twice because the first tme refers to the clean results while
    # the second time to the first poisoned results
        return pd.DataFrame([
            # hardcoded_results[model_name_]['0']['model_quality'][test_set_type_].drop(const.KEY_PIPELINE_NAME),
            hardcoded_results[model_name_]['0']['model_quality'][test_set_type_].drop(const.KEY_PIPELINE_NAME),
            hardcoded_results[model_name_]['1']['model_quality'][test_set_type_].drop(const.KEY_PIPELINE_NAME),
            hardcoded_results[model_name_]['2']['model_quality'][test_set_type_].drop(const.KEY_PIPELINE_NAME),
            hardcoded_results[model_name_]['3']['model_quality'][test_set_type_].drop(const.KEY_PIPELINE_NAME)]
        ).rename(
            lambda col: f'{model_name_}_{col}' if col not in const.INFO_KEY_LIST else col, axis='columns')

def build_expected_df_delta_self(hardcoded_results: HardcodedResultsType,
                                 model_name_, test_set_type_: common.TestSetType):
        # these columns are not used in delta.
        to_drop = col_to_drop_for_delta(hardcoded_results=hardcoded_results,
                                        model_name_=model_name_, test_set_type_=test_set_type_)

        diffs = [row['model_quality'][test_set_type_].drop(to_drop) -
                 hardcoded_results[model_name_]['0']['model_quality'][test_set_type_].drop(to_drop)
                 for row in [hardcoded_results[model_name_]['1'],
                             hardcoded_results[model_name_]['2'],
                             hardcoded_results[model_name_]['3']]]
        # now, we add the percentages once again.
        df = build_df_from_delta_diff(hardcoded_results=hardcoded_results,
                                      diffs_=diffs, model_name_=model_name_, test_set_type_=test_set_type_)
        return df

def build_expected_df_delta_ref(hardcoded_results: HardcodedResultsType,
                                model_name_: str, test_set_type_: common.TestSetType):
        # these columns are not used in delta.
        # Note that in this delta_ref we just the very same results.
        to_drop = col_to_drop_for_delta(hardcoded_results=hardcoded_results,
                                        model_name_=model_name_, test_set_type_=test_set_type_)

        diffs = [row['model_quality'][test_set_type_].drop(to_drop) - row['model_quality'][test_set_type_].drop(to_drop)
                 for row in [hardcoded_results[model_name_]['0'],
                            hardcoded_results[model_name_]['1'],
                             hardcoded_results[model_name_]['2'],
                             hardcoded_results[model_name_]['3']]]
        # now, we add the percentages once again.
        df = build_df_from_delta_diff(diffs_=diffs, model_name_=model_name_, test_set_type_=test_set_type_,
                                      hardcoded_results=hardcoded_results, )
        return df


def _merge(a, b):
        expected_df = a.merge(b, on=[const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES])
        expected_df = expected_df.reindex(sorted(expected_df.columns), axis=1)
        return expected_df


def _rearrange(to_arrange: dict):
        """
        We return a dict: [test_set_type: DataFrame]
        :return:
        """
        expected_ = {}
        # we now merge at model level, but first we have to separate at test set level,
        # so we do some arrangements.
        values_separated_test_set_type = {t: [] for t in common.TestSetType}
        for model_name, values_for_model in to_arrange.items():
            for test_set_type, values_for_test_set_type in values_for_model.items():
                values_separated_test_set_type[test_set_type].append(values_for_test_set_type)
        # now, we can merge
        for test_set_type, values_for_test_set_type in values_separated_test_set_type.items():
            # now we can merge
            expected_[test_set_type] = _merge(*values_for_test_set_type)
        return expected_


def test_analyzed_results():
    model_name_rf = 'rf'
    model_name_svm = 'svm'

    info_rf_0 = base.ExpInfo(perc_features=0, perc_points=0, pipeline_name=model_name_rf)
    info_rf_1 = base.ExpInfo(perc_features=0, perc_points=10, pipeline_name=model_name_rf)
    info_rf_2 = base.ExpInfo(perc_features=0, perc_points=20, pipeline_name=model_name_rf)
    info_rf_3 = base.ExpInfo(perc_features=0, perc_points=30, pipeline_name=model_name_rf)

    info_svm_0 = base.ExpInfo(perc_features=0, perc_points=0.0, pipeline_name=model_name_svm)
    info_svm_1 = base.ExpInfo(perc_features=0, perc_points=10, pipeline_name=model_name_svm)
    info_svm_2 = base.ExpInfo(perc_features=0, perc_points=20, pipeline_name=model_name_svm)
    info_svm_3 = base.ExpInfo(perc_features=0, perc_points=30, pipeline_name=model_name_svm)

    hardcoded_results = {
        model_name_rf: {
            '0': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET: build_quality_metrics(info_rf_0, 1),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_rf_0, 1)
                },
                'info': info_rf_0
            },
            '1': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_rf_1, 0.99),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_rf_1, 1)
                },
                'info': info_rf_1
            },
            '2': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_rf_2, 0.95),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_rf_2, 1)
                },
                'info': info_rf_2
            },
            '3': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_rf_3, 0.9),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_rf_3, 1)
                },
                'info': info_rf_3
            }
        },
        model_name_svm: {
            '0': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_svm_0, 0.97),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_svm_0, 1)
                },
                'info': info_svm_0
            },
            '1': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_svm_1, 0.85),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_svm_1, 1)
                },
                'info': info_svm_1
            },
            '2': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_svm_2, 0.85),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_svm_2, 1)
                },
                'info': info_svm_2
            },
            '3': {
                'model_quality': {
                    common.TestSetType.CLEAN_TEST_SET:build_quality_metrics(info_svm_3, 0.8),
                    common.TestSetType.CLEAN_TRAINING_SET: build_quality_metrics(info_svm_3, 1)
                },
                'info': info_svm_3
            }
        }
    }

    results = [
        common.CleanPoisonedOutputPair(
            pipeline_name=info_rf_0.pipeline_name,
            clean=common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_rf]['0']),
            poisoned=
            [
                # results for the random forest
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_rf]['1'],),
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_rf]['2']),
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_rf]['3']),
            ],
        ),
        common.CleanPoisonedOutputPair(
            pipeline_name=info_svm_0.pipeline_name,
            clean=common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_svm]['0']),
            poisoned=
            [
                # results for the svm
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_svm]['1']),
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_svm]['2'],),
                common.TrainSingleOutputMonolithicWithRep(**hardcoded_results[model_name_svm]['3']),
            ])
    ]

    got = exp.AnalyzedResultMonolithicModels.from_results(results_vanilla=results,
                                                          results_oracled=results, )

    ORACLE, VANILLA = 'oracle', 'vanilla'
    df_model_quality = {
        source: {m: {t: build_expected_df_model_quality(model_name_=m, test_set_type_=t,
                                                        hardcoded_results=hardcoded_results) for t in common.TestSetType}
                 for m in [model_name_rf, model_name_svm]}
        for source in [ORACLE, VANILLA]
    }
    df_delta_self = {
        source: {m: {t: build_expected_df_delta_self(model_name_=m, test_set_type_=t,
                                                        hardcoded_results=hardcoded_results) for t in common.TestSetType}
                 for m in [model_name_rf, model_name_svm]}
        for source in [ORACLE, VANILLA]
    }
    df_delta_ref = {m: {t: build_expected_df_delta_ref(model_name_=m, test_set_type_=t,
                                                        hardcoded_results=hardcoded_results) for t in common.TestSetType}
                    for  m in [model_name_rf, model_name_svm]}

    expected_model_quality_oracle = _rearrange(df_model_quality[ORACLE])
    expected_model_quality_vanilla = _rearrange(df_model_quality[VANILLA])
    expected_delta_self_oracle = _rearrange(df_delta_self[ORACLE])
    expected_delta_self_vanilla = _rearrange(df_delta_self[VANILLA])
    expected_delta_ref = _rearrange(df_delta_ref)

    expected = exp.AnalyzedResultMonolithicModels(model_quality_oracle=expected_model_quality_oracle,
                                                  model_quality_vanilla=expected_model_quality_vanilla,
                                                  delta_self_oracle=expected_delta_self_oracle,
                                                  delta_self_vanilla=expected_delta_self_vanilla,
                                                  delta_ref_vanilla_oracle=expected_delta_ref)
    for t in common.TestSetType:
        common_test.check_and_compare_df(got=got.model_quality_oracle[t], expected=expected_model_quality_oracle[t], )
        common_test.check_and_compare_df(got=got.model_quality_oracle[t], expected=expected.model_quality_oracle[t])
        common_test.check_and_compare_df(got=got.model_quality_vanilla[t], expected=expected.model_quality_vanilla[t])
        common_test.check_and_compare_df(got=got.delta_self_oracle[t], expected=expected.delta_self_oracle[t])
        common_test.check_and_compare_df(got=got.delta_self_vanilla[t], expected=expected.delta_self_vanilla[t])
        common_test.check_and_compare_df(got=got.delta_ref_vanilla_oracle[t], expected=expected.delta_ref_vanilla_oracle[t])

    # now, we export results.
    with tempfile.TemporaryDirectory() as temp_dir:
        export_config = exp.ExportConfigBaseModels(base_directory=temp_dir,
                                                                      exists_ok=True)
        got.export(config=export_config)

        expected_file_list = [f'{f}_{t.prefix()}.csv'
                              for f in [base.FILE_NAME_EXPORT_MONO_VANILLA_QUALITY, base.FILE_NAME_EXPORT_MONO_ORACLED_QUALITY,
                                        base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF,
                                        base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED]
                              for t in common.TestSetType]

        assert sorted(expected_file_list) == sorted(os.listdir(temp_dir))


def get_exp_from_dg(poisoning_generation_input: poisoning.PoisoningGenerationInput,
                    base_models, rep):
    dg = dg_test.get_dg(poisoning_generation_input=poisoning_generation_input)
    exp_ = exp.ExperimentMonolithicModels.from_dataset_generator(
        monolithic_models=base_models, dg=dg, repetitions=rep)
    return dg, exp_


def test_train_model_on_all_datasets():
    estimator = ('dt', tree.DecisionTreeClassifier())
    dg, exp_ = get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
        perc_data_points=[10.0, 15.0],
        performer=poisoning.PerformerLabelFlippingMonoDirectional(),
        selector=poisoning.SelectorRandom(),
        perform_info_kwargs={'from_label': 1, 'to_label': 0},
        perform_info_clazz=poisoning.PerformInfoMonoDirectional,
        selection_info_kwargs={},
        selection_info_clazz=poisoning.SelectionInfoEmpty
    ), base_models=[estimator], rep=2)
    result = exp_.train_model_on_all_datasets(estimator=estimator)
    # +1 because we want results on the clean dataset as well.
    assert len(result.poisoned) == len(dg)
    assert result.pipeline_name == estimator[0]


def test_do():
    estimators = [('dt', tree.DecisionTreeClassifier()),
                  ('rf', ensemble.RandomForestClassifier(n_estimators=10))]
    dg, exp_ = get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
        perc_data_points=[10.0, 15.0],
        performer=poisoning.PerformerLabelFlippingMonoDirectional(),
        selector=poisoning.SelectorRandom(),
        perform_info_kwargs={'from_label': 1, 'to_label': 0},
        perform_info_clazz=poisoning.PerformInfoMonoDirectional,
        selection_info_kwargs={},
        selection_info_clazz=poisoning.SelectionInfoEmpty
    ), base_models=estimators, rep=2)

    result = exp_.do()
    assert len(result[0]) == len(estimators)
    assert len(result[1]) == len(estimators)
