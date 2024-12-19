import pytest

from . import base, raw_dataset, raw_exp_abstract_test, raw_exp_monolithic_models, raw_poisoning
import experiments


@pytest.mark.parametrize('initial, expected', [
    (
            raw_exp_monolithic_models.ExportConfigExpBaseModelsRaw(exists_ok=True),
            experiments.ExportConfigBaseModels(exists_ok=True)
    )
])
def test_export_config_exp_base_models(initial, expected):
    raw_exp_abstract_test._test_export_config_exp_ensemble(initial=initial, expected=expected)


@pytest.mark.parametrize('initial', [
    raw_exp_monolithic_models.ExperimentMonolithicModelRaw(
        monolithic_models={
            'tree1':
                base.FuncPair(func_name='__sklearn.tree.DecisionTreeClassifier',
                              func_kwargs={}),
            'rf1':
                base.FuncPair(func_name='__sklearn.ensemble.RandomForestClassifier',
                              func_kwargs={}),
        },
        repetitions=5,
        base_output_directory='',
        dataset_exists_ok=True,
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 10.0, 15.0],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                selection_info_kwargs={'from_label': 0}
            )
        ),
    ),
])
def test_raw_exp_monolithic_models(initial: raw_exp_monolithic_models.ExperimentMonolithicModelRaw):
    got = raw_exp_abstract_test._test_raw_exp_basic(initial=initial)
    # here we check the name of the base models and the class name (of course)
    for base_model_name, base_model in initial.monolithic_models.items():
        found = False
        for got_base_model in got.monolithic_models:
            if base_model.func_name.split('.')[-1] in got_base_model[1].__class__.__name__ and \
                    base_model_name == got_base_model[0]:
                found = True
        assert found
