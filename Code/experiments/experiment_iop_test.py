import os.path
import tempfile
import typing

import numpy as np
import pandas as pd
from plotly import graph_objects as go
import pytest
from sklearn import cluster
import xarray as xr

import assignments
import const
import iops
import pipe
import poisoning

from . import base, dataset_generator_test as dg_test, experiment_iop as experiment


def simplest_pipeline(pipeline_name: str, keep_also_figures: bool = False) -> pipe.ExtPipeline:
    return pipe.ExtPipeline(name=pipeline_name,
                            steps_to_export=[0],
                            steps_to_evaluate=[0],
                            steps_to_figures=[0] if keep_also_figures else [],
                            steps=[pipe.Step('last',
                                             assignments.AssignmentRoundRobinBlind(N=3),
                                             output_col_names_pre=['assignments'])])


def _prepare_and_fake_pipelines(rep: int, info: base.ExpInfo, col_name: str,
                                add_also_iops: bool = False, add_also_figures: bool = False):
    poisoning_idx = np.concatenate([np.ones(100), np.zeros(100)])
    col_names = ['same', 'other']

    results = [
        # (
        pipe.ExtPipeline(name=info.pipeline_name,
                         steps=[
                             pipe.Step('distance', iops.IoPDistance(how=iops.IoPDistanceType.CLUSTERING,
                                                                    reverse_how=iops.ReverseType.SUBTRACT_BY_MAX,
                                                                    inner_kwargs={
                                                                        'clustering_clazz': cluster.KMeans,
                                                                        'direction': iops.Direction.FROM_OTHER_CLASS
                                                                    }),
                                       output_col_names_pre=col_names),
                             pipe.Step('last',
                                       assignments.AssignmentRoundRobinBlind(N=3),
                                       output_col_names_pre=[col_name])
                         ],
                         steps_to_evaluate=[1],
                         pre_assignment_idx=0,
                         steps_to_export=[0, 1] if add_also_iops else None,
                         steps_to_figures=[0, 1] if add_also_figures else None)
        # the first half will be poisoned, the second no.
        # np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1)
        # )
        for _ in range(rep)
    ]

    # set the result.
    for i in range(len(results)):
        # results[i][0].output_for_export[i] = (results[i][0].steps[-1], pipe.StepOutput(
        #     actual=(np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1), None),
        #     pre_aggregation_output=xr.DataArray(np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1),
        #                                         dims=('x', 'y'),
        #                                         coords={'y': [col_name]}),
        #     post_aggregation_output=xr.DataArray()
        # ))
        if add_also_iops or add_also_figures:
            results[i].output_for_export[0] = (
                results[i].steps[0],
                pipe.StepOutput(
                    actual=(
                        np.hstack([
                            np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1),
                            np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1)]),
                        None
                    ),
                    pre_aggregation_output=xr.DataArray(
                        np.hstack([
                            np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1),
                            np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1)]),
                        dims=('x', 'y'),
                        coords={'y': col_names}
                    ),
                    post_aggregation_output=xr.DataArray()
                )
            )

        results[i].output_for_export[1] = (results[i].steps[-1], pipe.StepOutput(
            actual=(np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1), None),
            pre_aggregation_output=xr.DataArray(np.concatenate([np.ones(100), np.zeros(100)]).reshape(-1, 1),
                                                dims=('x', 'y'),
                                                coords={'y': [col_name]}),
            post_aggregation_output=xr.DataArray()
        ))

    return results, poisoning_idx


@pytest.mark.parametrize('keep_also_iops, keep_also_figures', [(False, False), (True, True)])
def test_output_execute_pipeline_with_rep(keep_also_iops: bool, keep_also_figures: bool):
    rep = 5
    col_name = 'assignments'
    pipeline_name = 'p1'

    info = base.ExpInfo(pipeline_name=pipeline_name, perc_points=10.0, perc_features=0.0)

    pipelines, poisoning_idx = _prepare_and_fake_pipelines(rep=rep, info=info, col_name=col_name,
                                                           add_also_iops=keep_also_iops,
                                                           add_also_figures=keep_also_figures)

    got = experiment.ExecutePipelineWithRepOutput.from_results(results=pipelines, poisoning_idx=poisoning_idx,
                                                               info=info, keep_also_iops=keep_also_iops,
                                                               keep_also_figures=keep_also_figures)

    # the shape is (2 rows, 3 (pipe name, perc points, perc features) + 1 (poisoning) +
    # 1 (output column) * 2 (avg and std) )
    assert got.result.shape == (2, 3 + 1 + 1 * 2)
    assert np.all(got.result == pd.DataFrame([
        [info.pipeline_name, info.perc_points, info.perc_features, 0, 0.0, 0.0],
        [info.pipeline_name, info.perc_points, info.perc_features, 1, 1.0, 0.0]
    ], columns=[const.KEY_PIPELINE_NAME, const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES,
                const.KEY_ATTR_POISONED, f'{const.PREFIX_AVG}({col_name})', f'{const.PREFIX_STD}({col_name})']))

    # now check the risk as well.
    # accuracy, precision, recall. count_real, count_estimated + 3 (the info)
    assert len(got.result_risk) == 5 + 3

    # now check the IoPs.
    if keep_also_iops:
        if pipelines[0].steps_to_export is not None and len(pipelines[0].steps_to_export) > 0:
            assert len(got.iops_results) == rep
            # now we check that, for each pipeline,
            # the shape of the exported IoP matches the expected shape (i.e.,
            # 200 data points and the given number of columns, where this given number of columns
            # is the summation over the number of columns of each individual step in the pipeline.
            # In our case it is 2+1.)
            for iops_of_pipeline, p in zip(got.iops_results, pipelines):
                p: pipe.ExtPipeline = p
                assert iops_of_pipeline.shape == (200, 3)
                assert list(iops_of_pipeline.columns) == [f'{step.name}_{col}' for step in p.steps
                                                           for col in step.output_col_names_pre]

    if keep_also_figures:
        assert len(got.figures) == len(pipelines[0].steps_to_figures)
        for step_name, (exp_info, fig) in got.figures.items():
            assert step_name in pipelines[0].named_steps
            # assert isinstance(fig, matplotlib.figure.Figure)
            assert isinstance(fig, go.Figure)


def _prepare_fake_execution_with_rep_output(rep: int, infos: typing.List[base.ExpInfo], col_name,
                                            keep_also_iops: bool,
                                            keep_also_figures: bool) -> typing.List[experiment.ExecutePipelineWithRepOutput]:
    results = [_prepare_and_fake_pipelines(rep=rep, col_name=col_name, info=info,
                                           add_also_iops=keep_also_iops, add_also_figures=keep_also_figures)
               for info in infos]

    got_s = [experiment.ExecutePipelineWithRepOutput.from_results(results=result, info=info,
                                                                  poisoning_idx=poisoning_idx,
                                                                  keep_also_iops=keep_also_iops,
                                                                  keep_also_figures=keep_also_figures)
             for (result, poisoning_idx), info in zip(results, infos)]
    return got_s


@pytest.mark.parametrize('keep_also_iops, keep_also_figures', [(False, False), (True, True)])
def test_output_pipeline(keep_also_iops: bool, keep_also_figures: bool):
    rep = 2
    col_name = 'assignments'
    pipeline_name = 'p1'

    info_1 = base.ExpInfo(pipeline_name=pipeline_name, perc_points=10.0, perc_features=0.0)
    info_2 = base.ExpInfo(pipeline_name=pipeline_name, perc_points=11.0, perc_features=0.0)
    info_3 = base.ExpInfo(pipeline_name=pipeline_name, perc_points=12.0, perc_features=0.0)

    got_s = _prepare_fake_execution_with_rep_output(rep=rep, infos=[info_1, info_2, info_3], col_name=col_name,
                                                    keep_also_iops=keep_also_iops, keep_also_figures=keep_also_figures)

    got = experiment.ExecutePipelineOutputOnAll.from_results(got_s, keep_also_iops=keep_also_iops,
                                                             keep_also_figures=keep_also_figures)

    # the shape is (2 rows * | pipeline |, 3 (pipe name, perc points, perc features) + 1 (poisoning) +
    # 1 (output column) * 2 (avg and std) )
    assert got.result.shape == (2 * 3, 3 + 1 + 1 * 2)
    # 3 rows (3 pipeline), and 5 risk columns + 3 info
    assert got.result_risk.shape == (3, 5+3)
    if keep_also_iops:
        assert len(got.iops) == 3  # one for each poisoned data point
        for info, single_iop in got.iops:
            assert len(single_iop) == rep
    else:
        assert got.iops is None


@pytest.mark.parametrize('export_config', [
    experiment.ExportConfigIoP(export_also_iops=True, export_also_figures=True, export_png=True, export_html=True),
    experiment.ExportConfigIoP(export_also_iops=True, export_also_figures=True, export_png=True, export_html=False),
    experiment.ExportConfigIoP(export_also_iops=True, export_also_figures=True, export_png=False, export_html=True),
    experiment.ExportConfigIoP(export_also_iops=False, export_also_figures=False),
])
def test_export(export_config: experiment.ExportConfigIoP):
    rep = 3
    col_name = 'assignments'
    pipeline_names = ['p1', 'p2']

    # could be more parametric but...

    infos_p1 = [
        base.ExpInfo(pipeline_name=pipeline_names[0], perc_points=10.0, perc_features=0.0),
        base.ExpInfo(pipeline_name=pipeline_names[0], perc_points=12.0, perc_features=0.0),
        base.ExpInfo(pipeline_name=pipeline_names[0], perc_points=14.0, perc_features=0.0),
    ]

    infos_p2 = [
        base.ExpInfo(pipeline_name=pipeline_names[1], perc_points=10.0, perc_features=0.0),
        base.ExpInfo(pipeline_name=pipeline_names[1], perc_points=12.0, perc_features=0.0),
        base.ExpInfo(pipeline_name=pipeline_names[1], perc_points=14.0, perc_features=0.0),
    ]

    single_execs_p1 = _prepare_fake_execution_with_rep_output(rep=rep, infos=infos_p1, col_name=col_name,
                                                              keep_also_iops=True, keep_also_figures=True)
    aggregated_exec_p1 = experiment.ExecutePipelineOutputOnAll.from_results(single_execs_p1, keep_also_iops=True,
                                                                            keep_also_figures=True)

    single_execs_p2 = _prepare_fake_execution_with_rep_output(rep=rep, infos=infos_p2, col_name=col_name,
                                                              keep_also_iops=True, keep_also_figures=True)
    aggregated_exec_p2 = experiment.ExecutePipelineOutputOnAll.from_results(single_execs_p2, keep_also_iops=True,
                                                                            keep_also_figures=True)

    info_dict: typing.Dict[str, typing.Tuple[typing.List[base.ExpInfo], experiment.ExecutePipelineOutputOnAll]] = {
        pipeline_names[0]: (infos_p1, aggregated_exec_p1), pipeline_names[1]: (infos_p2, aggregated_exec_p2)}

    with tempfile.TemporaryDirectory() as temp_dir:

        export_config.base_directory = temp_dir

        analyzed_results = experiment.AnalyzedResultsIoP(results={pipeline_names[0]: aggregated_exec_p1,
                                                                  pipeline_names[1]: aggregated_exec_p2})
        analyzed_results.export(config=export_config)

        # now we begin to look that files are there
        # we begin by checking the aggregated results

        def _check_pipeline_names_in_dir(base_dir_: str):
            files = os.listdir(base_dir_)
            assert set(files) == set(f'{name}.csv' for name in pipeline_names)

        aggregated_dir_group = os.path.join(temp_dir, base.EXP_IOP_DIR_AGGREGATED)
        aggregated_dir_risk = os.path.join(temp_dir, base.EXP_IOP_DIR_RISK)

        for to_check in [aggregated_dir_risk, aggregated_dir_group]:
            assert os.path.exists(to_check) and os.path.isdir(to_check)
            _check_pipeline_names_in_dir(base_dir_=to_check)

        # assert os.path.exists(aggregated_dir_group) and os.path.isdir(aggregated_dir_group)
        # aggregated_result_files = os.listdir(aggregated_dir_group)
        # assert set(aggregated_result_files) == set(f'{name}.csv' for name in pipeline_names)

        # now we check the exported IoPs as well.
        if export_config.export_also_iops:

            iops_dir = os.path.join(temp_dir, base.EXP_IOP_DIR_INDIVIDUAL_IOP)
            assert os.path.join(iops_dir) and os.path.isdir(iops_dir)

            # now, there must be one subdirectory per pipeline.
            pipeline_dirs = os.listdir(iops_dir)
            assert set(pipeline_dirs) == set(pipeline_names)

            # now, in each directory there should be n_rep * n_poisoning files.
            for single_pipeline_dir in pipeline_dirs:
                got_files = os.listdir(os.path.join(os.path.join(temp_dir, iops_dir), single_pipeline_dir))
                # let's create the list of expected files.
                expected_files = set(f'{info.perc_points}_{info.perc_features}_{i}.csv'
                                     for info in info_dict[single_pipeline_dir][0] for i in range(rep))
                assert set(got_files) == expected_files

        if export_config.export_also_figures:
            # check that the plots directory exists.
            base_fig_dir = os.path.join(temp_dir, base.EXP_IOP_DIR_FIGURES)
            assert os.path.exists(base_fig_dir) and os.path.isdir(base_fig_dir)

            for pipeline_name in pipeline_names:
                current_fig_dir = os.path.join(base_fig_dir, pipeline_name)
                assert os.path.exists(current_fig_dir) and os.path.isdir(current_fig_dir)
                got_figures = os.listdir(current_fig_dir)
                # one figure for each step and percentage of poisoning
                expected_file_names = [f'{exp_info.mini_str()}_{step_name}'
                                       for figures_for_perc in info_dict[pipeline_name][1].figures
                                       for step_name, (exp_info, fig) in figures_for_perc.items()]

                expected_file_exts = []
                if export_config.export_png:
                    expected_file_exts.append('png')
                if export_config.export_html:
                    expected_file_exts.append('html')

                expected_file_names = [f'{file_name}.{ext}' for file_name in expected_file_names
                                       for ext in expected_file_exts]

                assert set(got_figures) == set(expected_file_names)


def get_exp_from_dg(poisoning_generation_input: poisoning.PoisoningGenerationInput,
                    pipes: typing.List[pipe.ExtPipeline], rep: int, keep_also_iops: typing.Optional[bool] = False,
                    keep_also_figures: typing.Optional[bool] = False):
    dg = dg_test.get_dg(poisoning_generation_input=poisoning_generation_input)
    exp = experiment.ExperimentIoP.from_dataset_generator(dg=dg, pipelines=pipes, repetitions=rep,
                                                          keep_also_iops=keep_also_iops,
                                                          keep_also_figures=keep_also_figures)
    return dg, exp


def test_create_exp_fails():
    # it fails because we create a pipeline requiring to save IoPS but there are
    # no important_steps to export.
    pipe_fail = simplest_pipeline(pipeline_name='p1')
    pipe_fail.important_steps = []

    with pytest.raises(ValueError):
        get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
            perc_data_points=[10, 15, 20],
            performer=poisoning.PerformerLabelFlippingMonoDirectional(),
            selector=poisoning.SelectorRandom(),
            perform_info_kwargs={'from_label': 1, 'to_label': 0},
            perform_info_clazz=poisoning.PerformInfoMonoDirectional,
            selection_info_kwargs={},
            selection_info_clazz=poisoning.SelectionInfoEmpty,
        ), pipes=[pipe_fail], rep=1, keep_also_iops=True, keep_also_figures=True)


@pytest.mark.parametrize('keep_also_iops, keep_also_figures', [(False, False), (True, True)])
def test_execute_single_pipeline_with_rep(keep_also_iops, keep_also_figures):
    pipeline_name = 'p1'
    rep = 5

    pipes = [simplest_pipeline(pipeline_name=pipeline_name, keep_also_figures=keep_also_figures)]

    dg, exp = get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
        perc_data_points=[10, 15, 20],
        performer=poisoning.PerformerLabelFlippingMonoDirectional(),
        selector=poisoning.SelectorRandom(),
        perform_info_kwargs={'from_label': 1, 'to_label': 0},
        perform_info_clazz=poisoning.PerformInfoMonoDirectional,
        selection_info_kwargs={},
        selection_info_clazz=poisoning.SelectionInfoEmpty,
    ), pipes=pipes, rep=rep, keep_also_iops=keep_also_iops, keep_also_figures=keep_also_figures)

    selected_info = base.ExpInfo(pipeline_name=pipeline_name, perc_points=10.0, perc_features=0.0)

    result = exp.execute_single_pipeline_with_rep(info=selected_info,
                                                  X=dg.X_train_clean, y=dg.y_train_clean,
                                                  pipeline=pipes[0],
                                                  poisoning_idx=np.random.default_rng().choice([0, 1], size=len(
                                                      dg.X_train_clean)))

    # there is not really much to check
    assert len(result.result) == 2
    assert np.all(result.result[const.KEY_PIPELINE_NAME] == selected_info.pipeline_name)
    assert np.all(result.result[const.KEY_PERC_DATA_POINTS] == selected_info.perc_points)
    assert np.all(result.result[const.KEY_PERC_FEATURES] == selected_info.perc_features)

    if keep_also_iops:
        assert len(result.iops_results) == rep
    else:
        assert result.iops_results is None

    if keep_also_figures:
        assert result.figures is not None
        # we do not execute with different poisoning configuration here.
        assert len(result.figures) == len(pipes[0].steps_to_figures) #* len(dg)
    else:
        assert result.figures is None


@pytest.mark.parametrize('keep_also_iops, keep_also_figures', [(False, False), (True, True)])
def test_execute_pipeline_on_all_poisoned(keep_also_iops, keep_also_figures):
    pipeline_name = 'p1'
    rep = 2

    pipes = [simplest_pipeline(pipeline_name=pipeline_name, keep_also_figures=keep_also_figures)]

    perc_points = [10.0, 15.0]
    perc_points_expanded = [10.0, 10.0, 15.0, 15.0]

    dg, exp = get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
        perc_data_points=perc_points,
        performer=poisoning.PerformerLabelFlippingMonoDirectional(),
        selector=poisoning.SelectorRandom(),
        perform_info_kwargs={'from_label': 1, 'to_label': 0},
        perform_info_clazz=poisoning.PerformInfoMonoDirectional,
        selection_info_kwargs={},
        selection_info_clazz=poisoning.SelectionInfoEmpty,
    ), pipes=pipes, rep=rep, keep_also_iops=keep_also_iops, keep_also_figures=keep_also_figures)

    result = exp.execute_pipeline_on_all_poisoned(pipeline=pipes[0])

    assert len(result.result) == len(dg) * 2
    assert np.all(result.result[const.KEY_PIPELINE_NAME] == pipeline_name)
    assert np.all(result.result[const.KEY_PERC_DATA_POINTS] == perc_points_expanded)
    assert np.all(result.result[const.KEY_PERC_FEATURES] == [0.0 for _ in range(len(perc_points_expanded))])

    if keep_also_iops:
        assert len(result.iops) == len(dg)
    else:
        assert result.iops is None
    if keep_also_figures:
        assert result.figures is not None
        # we do not execute with different poisoning configuration here.
        assert len(result.figures) == len(pipes[0].steps_to_figures) * len(dg)
    else:
        assert result.figures is None


@pytest.mark.parametrize('keep_also_iops, keep_also_figures', [(False, False), (True, True)])
def test_do(keep_also_iops, keep_also_figures):
    pipeline_names = ['p1', 'p2']
    pipes = [simplest_pipeline(pipeline_name=p_name) for p_name in pipeline_names]
    rep = 2

    _, exp = get_exp_from_dg(poisoning_generation_input=poisoning.PoisoningGenerationInput(
        perc_data_points=[1.0, 1.5],
        performer=poisoning.PerformerLabelFlippingMonoDirectional(),
        selector=poisoning.SelectorRandom(),
        perform_info_kwargs={'from_label': 1, 'to_label': 0},
        perform_info_clazz=poisoning.PerformInfoMonoDirectional,
        selection_info_kwargs={},
        selection_info_clazz=poisoning.SelectionInfoEmpty,
    ), pipes=pipes, rep=rep, keep_also_iops=keep_also_iops, keep_also_figures=keep_also_figures)

    result = exp.do()

    assert set(result.keys()) == set(pipeline_names)
