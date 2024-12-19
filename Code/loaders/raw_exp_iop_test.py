import dataclasses

import numpy as np
import pytest
import xarray as xr

import experiments
import pipe
from . import raw_exp_iop, raw_dataset, raw_exp_abstract_test, raw_poisoning, raw_pipe


@pytest.mark.parametrize('initial, expected', [
    (
        raw_exp_iop.ExportConfigExpIoPRaw(
            exists_ok=True,
            export_also_raw_results=False,
        ),
        experiments.ExportConfigIoP(
            exists_ok=True,
            export_also_iops=False
        )
    ),
    (
            raw_exp_iop.ExportConfigExpIoPRaw(
                exists_ok=True,
            ),
            experiments.ExportConfigIoP(
                exists_ok=True,
                export_also_figures=False,
                export_png=False,
                export_html=False
            )
    ),
    (
            raw_exp_iop.ExportConfigExpIoPRaw(
                exists_ok=True,
                export_also_figures=True,
                export_png=True,
                export_html=True
            ),
            experiments.ExportConfigIoP(
                exists_ok=True,
                export_also_figures=True,
                export_png=True,
                export_html=True
            )
    )
])
def test_export_config_exp_iop(initial: raw_exp_iop.ExportConfigExpIoPRaw,
                               expected: experiments.ExportConfigIoP):
    got = initial.parse()
    assert dataclasses.asdict(expected) == dataclasses.asdict(got)


def test_export_config_fail():
    with pytest.raises(ValueError):
        raw_exp_iop.ExportConfigExpIoPRaw(export_also_figures=True, export_html=False, export_png=False).parse()


@pytest.mark.parametrize('initial, expected', [
    (
        raw_exp_iop.ExperimentIoPRaw(
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
            export_config=raw_exp_iop.ExportConfigExpIoPRaw(
                exists_ok=True,
                export_also_raw_results=False
            ),
            base_output_directory='',
            repetitions=10,
            pipelines=[
                raw_pipe.PipelineRaw(
                    name='p1',
                    steps_to_evaluate=[0],
                    risk_idx=0,
                    steps=[
                        raw_pipe.StepRaw(name='s1', step_func_name='numpy.divide', output_col_names_pre=['Out']),
                    ],
                )
            ]
        ),
        experiments.ExperimentIoP(
            pipelines=[pipe.ExtPipeline(name='p1', steps=[pipe.Step(name='s1', step=np.divide)])],
            repetitions=10,
            # need to create one otherwise the constructor goes crazy
            poisoned_datasets=xr.Dataset(data_vars={'a': xr.DataArray(np.arange(1100).reshape((110, 10)),
                                                                      dims=('x', 'y'),
                                                                      coords={'y': [f'col{i}' for i in range(10)]})}),
            columns=[f'col{i}' for i in range(10)],
        )
    )
])
def test_raw_exp_iop(initial: raw_exp_iop.ExperimentIoPRaw,
                     expected: experiments.ExperimentIoP):


    # with tempfile.NamedTemporaryFile() as temp_file:
    #     # we create the dataset to be poisoned
    #     X, y = datasets.make_classification(110, 10)
    #     arr = np.hstack([X, y.reshape(-1, 1)])
    #     np.savetxt(temp_file.name, arr, delimiter=',',
    #                header=','.join([f'col{i}' for i in range(X.shape[1])] + [experiments.COORD_LABEL]))
    #
    #     initial.dataset_config.dataset_path = temp_file.name
    #
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         initial.base_output_directory = temp_dir
    #
    #         got = initial.parse()
    #
    #         files = os.listdir(temp_dir)
    #
    #         # what to expect here? the dataset
    #         assert files == [base.BASE_OUTPUT_DIR_DATASET]
    got = raw_exp_abstract_test._test_raw_exp_basic(initial=initial)
    # assert len(expected.pipelines) == len(initial.pipelines)
    assert len(got.pipelines) == len(expected.pipelines)
    for expected_p, got_p in zip(expected.pipelines, got.pipelines):
        assert expected_p.name == got_p.name
    assert expected.repetitions == got.repetitions


def test_raw_exp_iop_fail():
    # this is doomed to fail because we do not specify any
    # steps_to_evaluate.
    initial = raw_exp_iop.ExperimentIoPRaw(
            dataset_config_to_poison=raw_dataset.DatasetToPoisonRaw(
                dataset_path_training='',
                dataset_path_testing='',
                poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                    selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                    performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                    perc_data_points=[2.0, 5.0],
                    perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                    perform_info_kwargs={'from_label': 0, 'to_label': 1},
                    selection_info_clazz='_poisoning.SelectionInfoLabelMonoDirectionalRandom',
                    selection_info_kwargs={'from_label': 0}
                )
            ),
            dataset_exists_ok=False,
            export_config=raw_exp_iop.ExportConfigExpIoPRaw(
                exists_ok=True,
                export_also_raw_results=False
            ),
            base_output_directory='',
            repetitions=10,
            pipelines=[
                raw_pipe.PipelineRaw(
                    name='p1',
                    steps=[
                        raw_pipe.StepRaw(name='s1', step_func_name='numpy.divide', output_col_names_pre=['Out']),
                    ],
                )
            ]
        )

    with pytest.raises(ValueError):
        raw_exp_abstract_test._test_raw_exp_basic(initial=initial)
