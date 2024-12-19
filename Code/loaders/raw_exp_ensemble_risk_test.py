import pytest

from . import base, raw_exp_abstract_test, raw_dataset, raw_exp_ensemble_risk, raw_pipe, raw_poisoning
import experiments


@pytest.mark.parametrize('initial, expected', [
    (
            raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw(
                exists_ok=True,
            ),
            experiments.ExportConfigExpEnsembleRisk(
                exists_ok=True,
            )
    )
])
def test_export_config_exp_ensemble(initial: raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw,
                                    expected: experiments.ExportConfigExpEnsembleRisk):
    raw_exp_abstract_test._test_export_config_exp_ensemble(initial=initial, expected=expected)


@pytest.mark.parametrize('initial', [
    # (
    raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw(
        monolithic_model=base.FuncPair(
            func_name='__sklearn.tree.DecisionTreeClassifier',
            func_kwargs={'min_samples_leaf': 1},
        ),
        # base_model_name='__sklearn.tree.DecisionTreeClassifier',
        # base_model_kwargs={'min_samples_leaf': 1},
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
        export_config=raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw(
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
    # now we also do one test where we have some baselines
    raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw(
        monolithic_model=base.FuncPair(
            func_name='__sklearn.tree.DecisionTreeClassifier',
            func_kwargs={'min_samples_leaf': 1},
        ),
        # base_model_name='__sklearn.tree.DecisionTreeClassifier',
        # base_model_kwargs={'min_samples_leaf': 1},
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
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        export_config=raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw(
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
        know_all_pipelines=[
            raw_pipe.PipelineRaw(
                risk_idx=0,
                name='BASELINE(p1)',
                steps=[
                    raw_pipe.StepRaw(name='s1', step_func_name='__assignments.AssignmentRoundRobinBlind',
                                     step_func_kwargs={'N': 3})
                ],
            )
        ],
        base_output_directory=''
    ),
])
def test_raw_exp_ensemble(initial: raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw, ):
    raw_exp_abstract_test._test_raw_exp_ensemble(initial=initial)
