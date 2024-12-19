import pytest

from . import base, raw_dataset, raw_exp_ensemble_plain_advanced, raw_pipe, raw_poisoning, raw_exp_abstract_test
import experiments


@pytest.mark.parametrize('initial, expected', [
    (
            raw_exp_ensemble_plain_advanced.ExportConfigExpEnsemblePlainAdvancedRaw(
                exists_ok=True,
            ),
            experiments.ExportConfigExpEnsemblePlainAdvanced(
                exists_ok=True,
            )
    )
])
def test_export_config_exp_ensemble(initial: raw_exp_ensemble_plain_advanced.ExportConfigExpEnsemblePlainAdvancedRaw,
                                    expected: experiments.ExportConfigExpEnsemblePlainAdvanced):
    raw_exp_abstract_test._test_export_config_exp_ensemble(initial=initial, expected=expected)


@pytest.mark.parametrize('initial', [
    # (
    raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw(
        monolithic_model=base.FuncPair(
            func_name='__sklearn.tree.DecisionTreeClassifier',
            func_kwargs={'min_samples_leaf': 1},
        ),
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 10.0],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        export_config=raw_exp_ensemble_plain_advanced.ExportConfigExpEnsemblePlainAdvancedRaw(
            exists_ok=True,
        ),
        repetitions=5,
        pipelines=[
            raw_pipe.PipelineRaw(
                risk_idx=0,
                name='p1',
                steps=[
                    raw_pipe.StepRaw(name='s1', step_func_name='__assignments.AssignmentRoundRobinBlind',
                                     step_func_kwargs={'N': 3})
                ],
            )
        ],
        base_output_directory=''
    ),
    # )
])
def test_raw_exp_ensemble(initial: raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw, ):
    raw_exp_abstract_test._test_raw_exp_ensemble(initial=initial)
