import os
import tempfile
import typing

import pytest
from sklearn import datasets, model_selection

from . import (base, raw_dataset, raw_exp_ensemble_plain_advanced,
               raw_exp_ensemble_risk, raw_exp_iop, raw_pipe, raw_poisoning, \
    raw_dataset_test,
               top)


@pytest.mark.parametrize('config', [
    top.TopLevelExpIop(
        base_output_directory='',
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 7.5, 8.5],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                # selection_info_kwargs={'from_label': 0}
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        repetitions=2,
        export_config=raw_exp_iop.ExportConfigExpIoPRaw(
            exists_ok=True,
            export_also_raw_results=False,
        ),
        pipelines=[
            raw_pipe.PipelineRaw(name='p1',
                                 steps_to_evaluate=[0],
                                 steps=[
                                     raw_pipe.StepRaw('last',
                                                      '__assignments.AssignmentRoundRobinBlind',
                                                      output_col_names_pre=['assignment'],
                                                      step_func_kwargs={'N': 3})
                                 ])
        ]
    ),
    top.TopLevelExpEnsemblePlainAdvanced(
        # base_model_name='__sklearn.ensemble.RandomForestClassifier',
        monolithic_model=base.FuncPair(func_name='__sklearn.ensemble.RandomForestClassifier'),
        base_output_directory='',
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 7.5, 8.1],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        export_config=raw_exp_ensemble_plain_advanced.ExportConfigExpEnsemblePlainAdvancedRaw(
            exists_ok=True
        ),
        repetitions=2,
        pipelines=[
            raw_pipe.PipelineRaw(name='p1',
                                 steps=[
                                     raw_pipe.StepRaw('last',
                                                      '__assignments.AssignmentRoundRobinBlind',
                                                      step_func_kwargs={'N': 3})
                                 ])
        ]
    ),
    top.TopLevelExpEnsembleRisk(
        # base_model_name='__sklearn.ensemble.RandomForestClassifier',
        monolithic_model=base.FuncPair(func_name='__sklearn.ensemble.RandomForestClassifier'),
        base_output_directory='',
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 7.5],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        export_config=raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw(
            exists_ok=True
        ),
        repetitions=2,
        pipelines=[
            raw_pipe.PipelineRaw(name='p1',
                                 steps_to_evaluate=[0],
                                 risk_idx=0,
                                 steps=[
                                     raw_pipe.StepRaw('last',
                                                      '__assignments.AssignmentRoundRobinBlind',
                                                      output_col_names_pre=['assignment'],
                                                      step_func_kwargs={'N': 3})
                                 ])
        ]
    ),
    # one test where we also have some baselines.
    top.TopLevelExpEnsembleRisk(
        # base_model_name='__sklearn.ensemble.RandomForestClassifier',
        monolithic_model=base.FuncPair(func_name='__sklearn.ensemble.RandomForestClassifier'),
        base_output_directory='',
        dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
            dataset_path_training='',
            dataset_path_testing='',
            poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[5.0, 7.5],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoEmpty',
                selection_info_kwargs={}
            )
        ),
        dataset_exists_ok=True,
        export_config=raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw(
            exists_ok=True
        ),
        repetitions=2,
        pipelines=[
            raw_pipe.PipelineRaw(name='p1',
                                 steps_to_evaluate=[0],
                                 risk_idx=0,
                                 steps=[
                                     raw_pipe.StepRaw('last',
                                                      '__assignments.AssignmentRoundRobinBlind',
                                                      output_col_names_pre=['assignment'],
                                                      step_func_kwargs={'N': 3})
                                 ])
        ],
        know_all_pipelines=[
            raw_pipe.PipelineRaw(name='BASELINE(p1)',
                                 steps_to_evaluate=[0],
                                 risk_idx=0,
                                 steps=[
                                     raw_pipe.StepRaw('last',
                                                      '__assignments.AssignmentRoundRobinBlind',
                                                      output_col_names_pre=['assignment'],
                                                      step_func_kwargs={'N': 3})
                                 ])
        ]
    ),
])
def test_exp(config: typing.Union[top.TopLevelExpIop, top.TopLevelExpEnsembleRisk, top.TopLevelExpEnsemblePlainAdvanced]):
    X, y = datasets.make_classification()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    with tempfile.NamedTemporaryFile() as training_file, tempfile.NamedTemporaryFile() as testing_file:
        config.dataset_config.dataset_path_training = training_file.name
        config.dataset_config.dataset_path_testing = testing_file.name

        raw_dataset_test.X_y_to_csv(X_train, y_train, training_file.name)
        raw_dataset_test.X_y_to_csv(X_test, y_test, testing_file.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_output_directory = temp_dir
            config.do()

            # just ensure it is not empty
            assert len(os.listdir(temp_dir)) > 0

        # now, we do a very simple test: we re-do the same work but forbidding
        # result overriding.
        config.export_config.exists_ok = False
        with tempfile.TemporaryDirectory() as temp_dir:
            # we need to create the specific subdirectory as well.
            output_dir = os.path.join(os.path.abspath(temp_dir), base.BASE_OUTPUT_DIR_OUTPUT)
            os.makedirs(output_dir, exist_ok=True)
            # but we pass as input the base dir, because the code itself will add base.BASE_OUTPUT_DIR_OUTPUT
            # at the end.
            config.base_output_directory = temp_dir

            with pytest.raises(ValueError):
                config.do()


@pytest.mark.parametrize('config', [
    top.TopLevelDataset(
        base_output_directory='',
        dataset_path_training='',
        dataset_path_testing='',
        exists_ok=True,
        poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
            selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
            performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
            perc_data_points=[5.0, 7.5],
            perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
            perform_info_kwargs={'from_label': 0, 'to_label': 1},
            selection_info_clazz='_poisoning.SelectionInfoEmpty',
            selection_info_kwargs={}
        )
    )
])
def test_dg(config: top.TopLevelDataset):
    X, y = datasets.make_classification()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    with tempfile.NamedTemporaryFile() as training_file, tempfile.NamedTemporaryFile() as testing_file:
        config.dataset_path_training = training_file.name
        config.dataset_path_testing = testing_file.name

        raw_dataset_test.X_y_to_csv(X_train, y_train, training_file.name)
        raw_dataset_test.X_y_to_csv(X_test, y_test, testing_file.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_output_directory = temp_dir
            config.do()
