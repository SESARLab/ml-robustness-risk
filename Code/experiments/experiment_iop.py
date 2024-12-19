import copy
import dataclasses
import os.path
import typing
import warnings

import joblib

# from matplotlib import figure, pyplot as plt
from plotly import subplots
from plotly import graph_objects as go
from plotly.graph_objects import scatter

import numpy as np
import pandas as pd
import xarray as xr

import const
import pipe
import utils_exp_post
from . import base, dataset_generator


def xr_to_df(data_array: xr.DataArray) -> pd.DataFrame:
    # could use data_array.to_df, but it does not really work
    # as I would
    if len(data_array.shape) > 2:
        raise ValueError('Could output 2d xr.DataArray only')

    arr = data_array.to_numpy()
    return pd.DataFrame(arr, columns=data_array.coords['y'].values.tolist())


@dataclasses.dataclass
class ExecutePipelineWithRepOutput:
    """
    contains the result of the execution of a single pipeline
    multiple times on the same (poisoned) dataset (i.e., with a single percentage of poisoning).
    """
    info: base.ExpInfo
    # 2-row pd.DataFrame containing the average of the required pipeline steps without (row 0)
    # and with poisoning (row 1).
    # Note that this is different from clean or poisoned dataset. Here, we work at data point level.
    result: pd.DataFrame
    # averaged risk quality
    result_risk: pd.Series
    # the figure plotting the results. The dict maps between the step name and the (exp info, figure)
    figures: typing.Optional[typing.Dict[str, typing.Tuple[base.ExpInfo, go.Figure]]] = dataclasses.field(
        default=None)
    iops_results: typing.Optional[typing.List[pd.DataFrame]] = dataclasses.field(default=None)

    @staticmethod
    def from_results(
            results: typing.List[pipe.ExtPipeline], poisoning_idx: np.ndarray,
            info: base.ExpInfo, keep_also_iops: bool = False, keep_also_figures: bool = False, ):
        # for each repetition, we create a pd.DataFrame
        # with the output of the pipeline and an additional column "poisoning_idx".

        def _inner_func(result_: pipe.ExtPipeline):
            mean_l = []
            std_l = []

            for step_to_evaluate_ in result_.steps_to_evaluate:
                df = utils_exp_post.data_array_to_df(result_.output_for_export[step_to_evaluate_][1].get_pre_and_post_as_xr())
                df[const.KEY_ATTR_POISONED] = poisoning_idx
                grouped = df.groupby([const.KEY_ATTR_POISONED])

                # an individual result of mean() and std() is a DataFrame looking like this:
                #               col_name
                #  Poisoning
                #   0               <val>
                #   1               <val>
                mean_l.append(grouped.mean())
                std_l.append(grouped.std())

            # retrieve the risk quality. It is a pd.Series
            risk_quality, _ = base.extract_and_evaluate_risk(p=result_, poisoning_info=poisoning_idx)

            # here we just need to join the different mean(s) and std(s)
            # so that later, when we average their value over the individual repetitions,
            # we can do that with ease.
            join_mean = utils_exp_post.just_merge_repeatedly(mean_l)
            join_std = utils_exp_post.just_merge_repeatedly(std_l)

            return join_mean, join_std, risk_quality

        # we retrieve the individual avg and mean for each required step in each pipeline.
        raw_results = joblib.Parallel(n_jobs=len(results))(joblib.delayed(_inner_func)(p) for p in results)

        # deconstruct results
        agg_results_mean = []
        agg_results_std = []
        agg_results_risk = []

        for single_result_mean, single_result_std, single_risk_quality in raw_results:
            agg_results_mean.append(single_result_mean)
            agg_results_std.append(single_result_std)
            agg_results_risk.append(single_risk_quality)

        # now we compute the average of the retrieved mean and std,
        # that is, their average value over the different executions (repetitions) of this pipeline.
        # now here we have a pd.DataFrame containing the mean, one for each execution.
        # each item we concat is a 2-row pd.DataFrame, containing the mean for poisoned = 0
        # and poisoned = 1
        # i.e.,
        # agg_results_mean[0] = pd.DataFrame([[0, ...], [1, ...]])
        # where 0 and 1 are the index of the pd.DataFrame, and the remaining values are the actual averaged
        # values output of the pipeline. We do reset_index() so the index (containing whether poisoning or not)
        # is set as a normal column. This last step is not necessary, just a precaution.
        concat_mean = pd.concat(agg_results_mean).reset_index()
        concat_std = pd.concat(agg_results_std).reset_index()
        # we just we have one pd.Series for each repetition.
        concat_risk = pd.DataFrame(agg_results_risk)

        # now, we do another group_by on poisoned and poisoned to retrieve
        # the final avg. The two "mean()" are correct.
        mean_mean = concat_mean.groupby([const.KEY_ATTR_POISONED]).mean()
        mean_std = concat_std.groupby([const.KEY_ATTR_POISONED]).mean()
        mean_risk = concat_risk.mean()

        # now we build the 2-row pd.DataFrame containing the values that we retrieved, the indication of
        # poisoned or not, and the ExpInfo. We are going to do a join, but we can't do a join
        # where the columns have the same name, so we have to rename them first, then we join on the index.
        mean_mean = mean_mean.rename(lambda col: f'{const.PREFIX_AVG}({col})', axis='columns')
        mean_std = mean_std.rename(lambda col: f'{const.PREFIX_STD}({col})', axis='columns')
        # by doing reset_index() the column containing information on poisoning is placed first.
        joined = mean_mean.join(mean_std, validate='one_to_one').reset_index()

        # now let's add the poisoning info.
        joined = info.prepend_to(joined)
        mean_risk = info.prepend_to(mean_risk)

        iops_result = None

        # if we want to keep also the results of the IoPs and the pipeline did have some
        # output_for_export.
        if keep_also_iops and len(results[0].steps_to_export) > 0:

            def _inner(single_pipeline_: pipe.ExtPipeline):
                output_df_l = []
                # collects the IoPs we are interested in export.
                output_l = [(single_pipeline_.output_for_export[i][0],
                             utils_exp_post.data_array_to_df(single_pipeline_.output_for_export[i][1].get_pre_and_post_as_xr()))
                            for i in single_pipeline_.steps_to_export]

                # now we rename the column of each pd.DataFrame prepending the name
                # of the step.
                for i_ in range(len(output_l)):
                    step = output_l[i_][0]
                    output_df_l.append(output_l[i_][1].rename(lambda col: f'{step.name}_{col}', axis='columns'))
                # now we can finally join everything together.
                return utils_exp_post.just_merge_repeatedly(output_df_l)

            iops_result = joblib.Parallel(n_jobs=len(results))(joblib.delayed(_inner)(p) for p in results)

        figures = None
        if keep_also_figures:
            # indexing fails if poisoning_idx is not considered as a boolean array
            poisoning_idx_ = poisoning_idx.astype(bool)
            figures = {}
            # we must deconstruct the results to have the same set of results for each step across
            # repetition, i.e., to print the same step output in the same figure
            # Note: the pipelines are just copy of each other, so we can safely access only the first one.
            deconstructed_for_figures = {k: [] for k in results[0].steps_to_figures}
            for step_to_plot_idx in results[0].steps_to_figures:
                for pipeline_rep in results:
                    deconstructed_for_figures[step_to_plot_idx].append(
                        pipeline_rep.output_for_export[step_to_plot_idx][1])

            for step_to_plot_idx, outputs in deconstructed_for_figures.items():

                # title of the plot (individual subplot)
                pipeline_and_step_name = f'{results[0].name}_{results[0].steps[step_to_plot_idx].name})'

                # fig, axs = plt.subplots(nrows=len(results), ncols=2, figsize=(15, 15),
                #                         layout='constrained')
                fig = subplots.make_subplots(rows=len(results), cols=2, start_cell='top-left',
                                             # subplot_titles=[f'{pipeline_and_step_name}_{when}, rep: {rep}'
                                             #                 for rep in range(len(results))
                                             #                 for when in ['PRE', 'POST']],
                                             column_widths=[0.5, 0.5],  # vertical_spacing=(1 / (len(results) - 1))
                                             vertical_spacing=0.075
                                             )

                for i, output in enumerate(outputs):

                    # in plotly rows and columns cont starts at 1
                    row_idx = i + 1

                    output_pre: xr.DataArray = output.pre_aggregation_output
                    output_post: xr.DataArray = output.post_aggregation_output

                    # whether to do the first plot.
                    do_first = True

                    if len(output_pre.shape) == 1 or output_pre.shape[1] == 1:
                        # if output is 1d...
                        X = output_pre.to_series()
                        y = np.zeros(len(output_pre))

                        x_title = output_pre.coords['y'].values[0]
                        y_title = 'dummy'
                    else:
                        # if output is 2d
                        if output_pre.shape[1] != 2:
                            warnings.warn('Plot has more than 2 dimensions. Skip.')
                            do_first = False
                        else:
                            X = output_pre[:, 0]
                            y = output_pre[:, 1]

                            x_title = output_pre.coords['y'].values[0]
                            y_title = output_pre.coords['y'].values[1]

                    if do_first:

                        # low alpha so that we can easily look for overlaps.
                        # axs[i, 0].scatter(x=X[~poisoning_idx_], y=y[~poisoning_idx_], color='green', alpha=0.4,
                        #                   label='Non-Poisoned', marker='o')
                        # axs[i, 0].scatter(x=X[poisoning_idx_], y=y[poisoning_idx_], color='red', alpha=0.4,
                        #                   label='Poisoned', marker='^')
                        #
                        # axs[i, 0].set_title(f'{pipeline_and_step_name}_PRE, repetition: {i}')
                        # axs[i, 0].legend()
                        kwargs_scatter_non_poisoned = {'showlegend': row_idx == 1}
                        kwargs_scatter_poisoned = {'showlegend': row_idx == 1}
                        if row_idx == 1:  # so it is not repeated every time.
                            kwargs_scatter_non_poisoned['name'] = 'Non-Poisoned'
                            kwargs_scatter_poisoned['name'] = 'Poisoned'

                        fig.add_trace(go.Scatter(x=X[~poisoning_idx_], y=y[~poisoning_idx_], mode='markers',
                                                 marker=scatter.Marker(color='green', opacity=0.4, symbol='circle'),
                                                 **kwargs_scatter_non_poisoned
                                                 ), row=row_idx, col=1)
                        fig.add_trace(go.Scatter(x=X[poisoning_idx_], y=y[poisoning_idx_], mode='markers',
                                                 marker=scatter.Marker(color='red', opacity=0.4, symbol='triangle-up'),
                                                 **kwargs_scatter_poisoned
                                                 ), row=row_idx, col=1)
                        fig.update_xaxes(title_text=x_title, title_font={'size': 10}, row=row_idx, col=1)
                        fig.update_yaxes(title_text=y_title, title_font={'size': 10}, row=row_idx, col=1)

                    if not output_pre.equals(output_post) and do_first:
                        # plot also the second column, with the "binarized" risk.
                        risky_idx = np.argwhere(output_post.values == 1).flatten()
                        non_risky_idx = np.argwhere(output_post.values == 0).flatten()

                        poisoned_idx = np.argwhere(poisoning_idx_ == True).flatten()
                        non_poisoned_idx = np.argwhere(poisoning_idx_ == False).flatten()
                        risky_and_poisoned_idx = np.intersect1d(risky_idx, poisoned_idx)

                        other_idx = np.arange(len(y))[~risky_and_poisoned_idx]

                        fig.add_trace(
                            go.Scatter(x=X[other_idx],
                                       y=y[other_idx],
                                       marker=scatter.Marker(color='green', opacity=0.3, symbol='circle'),
                                       mode='markers',
                                       name='Other'), row=row_idx, col=2)

                        fig.add_trace(go.Scatter(x=X[risky_and_poisoned_idx], y=y[risky_and_poisoned_idx],
                                                 mode='markers',
                                                 marker=scatter.Marker(color='red', opacity=0.4, symbol='triangle-up'),
                                                 name='Poisoned, risk=1'), row=row_idx, col=2)

                fig.update_layout(  # xaxis_range=[0, 1], yaxis_range=[0, 1],
                    margin={'t': 70, 'l': 0, 'r': 0, 'b': 0},
                    title_text=f'{pipeline_and_step_name}[PRE and POST]',
                    legend={'yanchor': 'top', 'y': 1.05, 'xanchor': 'left', 'x': 0.01, 'orientation': 'h',
                            'font': {'size': 10}},
                    width=650, height=800)
                figures[results[0].steps[step_to_plot_idx].name] = (info, fig)

        return ExecutePipelineWithRepOutput(info=info, result=joined, iops_results=iops_result,
                                            result_risk=mean_risk, figures=figures)


@dataclasses.dataclass
class ExecutePipelineOutputOnAll:
    """
    Results of executing the pipeline on a poisoned dataset varying the percentage of poisoning.
    """
    result: pd.DataFrame
    result_risk: pd.DataFrame
    # one item for each percentage of poisoning. So, each item refers to a specific poisoning config, and then
    # maps the step (name) to the figure and corresponding exp info.
    figures: typing.Optional[
        typing.List[typing.Dict[str, typing.Tuple[base.ExpInfo, go.Figure]]]] = dataclasses.field(
        default=None)
    iops: typing.Optional[typing.List[typing.Tuple[base.ExpInfo, typing.List[pd.DataFrame]]]] = dataclasses.field(
        default=None)

    @staticmethod
    def from_results(results: typing.List[ExecutePipelineWithRepOutput],
                     keep_also_iops: bool = False, keep_also_figures: bool = False, ):
        # what we have to do is to "compose"
        # the overall result pd.DataFrame by just placing the results one after the
        # other.
        dfs = []
        # a list of pd.Series
        risk_s = []
        iops: typing.List[typing.Tuple[base.ExpInfo, typing.List[pd.DataFrame]]] = [] if keep_also_iops else None
        figures: typing.List[typing.Dict[str, typing.Tuple[base.ExpInfo, go.Figure]]] = [] if (
            keep_also_figures) else None

        for single_result in results:
            dfs.append(single_result.result)
            risk_s.append(single_result.result_risk)
            if keep_also_iops:
                iops.append((single_result.info, single_result.iops_results))
            if keep_also_figures:
                figures.append(single_result.figures)

        # that's it!
        # what we do there is appending the pd.DataFrame one below the other. We then drop the index.
        # It is not necessary to add the pipeline name, percentage of poisoning, and so on
        # because they are already added in the pd.DataFrame we are concatenating.
        return ExecutePipelineOutputOnAll(result=pd.concat(dfs).reset_index(drop=True),
                                          result_risk=pd.DataFrame(risk_s),
                                          iops=iops, figures=figures)


@dataclasses.dataclass
class ExportConfigIoP(base.AbstractExportConfigWithDirectory):
    export_also_iops: bool = dataclasses.field(default=False)
    export_also_figures: bool = dataclasses.field(default=False)
    export_png: bool = dataclasses.field(default=False)
    export_html: bool = dataclasses.field(default=False)

    def __post_init__(self):
        if self.export_also_figures and not self.export_png and not self.export_html:
            raise ValueError('if export_also_figures then at least 1 format must be selected')


@dataclasses.dataclass
class AnalyzedResultsIoP:
    """
    Wrapper class containing the results of executing multiple pipelines.
    """
    results: typing.Dict[str, ExecutePipelineOutputOnAll]

    @staticmethod
    def from_results(results: typing.Dict[str, ExecutePipelineOutputOnAll]):
        return AnalyzedResultsIoP(results=results)

    def export(self, config: ExportConfigIoP):
        if config.base_directory is None:
            return

        # we need to create two directories. In the first one
        # we save the aggregated data, i.e., those with average and so on.
        # in the second one we eventually export the individual IoPs.
        dir_aggregated_group = os.path.join(config.base_directory, base.EXP_IOP_DIR_AGGREGATED)
        dir_aggregated_risk = os.path.join(config.base_directory, base.EXP_IOP_DIR_RISK)
        dir_plots = os.path.join(config.base_directory, base.EXP_IOP_DIR_FIGURES)

        to_create = [dir_aggregated_group, dir_aggregated_risk]
        if config.export_also_figures:
            to_create.append(dir_plots)

        for to_create in to_create:
            os.makedirs(to_create, exist_ok=config.exists_ok)

        for pipeline_name, pipeline_results in self.results.items():
            # this is the aggregated result (i.e., grouped)
            pipeline_results.result.to_csv(os.path.join(dir_aggregated_group, f'{pipeline_name}.csv'), index=False)
            # this is the risk results
            pipeline_results.result_risk.to_csv(os.path.join(dir_aggregated_risk, f'{pipeline_name}.csv'), index=False)

            # now let's see if we need to export IoPs as well.
            if config.export_also_iops:
                if pipeline_results.iops is None or len(pipeline_results.iops) == 0:
                    raise ValueError(f'Requested to export also IoPs but no IoP found for {pipeline_name}')

                dir_individual = os.path.join(config.base_directory, base.EXP_IOP_DIR_INDIVIDUAL_IOP)

                # we first create the directory holding the different executions of the IoPs
                # in this pipeline.
                base_iop_pipeline_dir = os.path.join(dir_individual, pipeline_name)
                os.makedirs(base_iop_pipeline_dir, exist_ok=config.exists_ok)

                # now pipeline_results.iops is a list of pairs (ExpInfo, list[pd.DataFrame])
                # Each pair contains **all** the results of this pipeline execution.
                # In particular, we have the percentage of poisoning and the set of results
                # retrieved under this percentage: we have set of results because the pipeline is executed
                # multiple times for each percentage.
                for all_iop_of_this_pipeline in pipeline_results.iops:
                    exp_info: base.ExpInfo = all_iop_of_this_pipeline[0]
                    actual_results: typing.List[pd.DataFrame] = all_iop_of_this_pipeline[1]
                    for i, actual_result in enumerate(actual_results):
                        # the file is called perc_points_perc_features_rep-number.csv
                        actual_result.to_csv(os.path.join(base_iop_pipeline_dir, f'{exp_info.mini_str()}_{i}.csv'),
                                             index=False)

            if config.export_also_figures:
                # dir_html = os.path.join(dir_plots, base.EXP_IOP_DIR_FIGURES_HTML)
                # dir_png = os.path.join(dir_plots, base.EXP_IOP_DIR_FIGURES_PNG)

                # create a directory for each pipeline.
                pipeline_fig_dir = os.path.join(dir_plots, pipeline_name)
                os.makedirs(pipeline_fig_dir, exist_ok=config.exists_ok)
                # print(f'exporting figures: {pipeline_results.figures}')
                for figures in pipeline_results.figures:
                    for step_name, (exp_info, figure_) in figures.items():
                        # figure_.savefig(os.path.join(pipeline_fig_dir, f'{exp_info.mini_str()}_{step_name}.png'))
                        # print(f'exporting {pipeline_fig_dir}+{exp_info.mini_str()}_{step_name}.png')
                        if config.export_html:
                            figure_.write_html(os.path.join(pipeline_fig_dir, f'{exp_info.mini_str()}_{step_name}.html'),
                                               include_plotlyjs='cdn',)
                                           #include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js',)
                        if config.export_png:
                            figure_.write_image(os.path.join(pipeline_fig_dir, f'{exp_info.mini_str()}_{step_name}.png'),
                                            scale=6)


class ExperimentIoP(base.AbstractExperiment):

    def __init__(self, repetitions: int,
                 poisoned_datasets: xr.Dataset,
                 pipelines: typing.List[pipe.ExtPipeline],
                 columns: typing.Optional[typing.List[str]] = None,
                 keep_also_iops: typing.Optional[bool] = False,
                 keep_also_figures: typing.Optional[bool] = False,
                 # export_png: typing.Optional[bool] = False,
                 # export_html: typing.Optional[bool] = False,
                 ):
        super().__init__(repetitions=repetitions, clean_dataset_attrs={},
                         poisoned_datasets=poisoned_datasets, columns=columns)
        self.pipelines = pipelines
        self.results: typing.Dict[str, typing.List[xr.Dataset]] = dict()
        self.keep_also_iops = keep_also_iops or False
        self.keep_also_figures = keep_also_figures or False
        # self.export_html = export_html or False
        # self.export_png = export_png or False

        base.check_unique_pipeline_names(self.pipelines)

        if keep_also_iops:
            for pipeline in pipelines:
                if len(pipeline.important_steps) == 0:
                    raise ValueError('You require to export IoPs but there are no important steps '
                                     f'in pipeline "{pipeline.name}"')

    @property
    def analysis_class(self) -> typing.Type[AnalyzedResultsIoP]:
        return AnalyzedResultsIoP

    @staticmethod
    def from_dataset_generator(dg: dataset_generator.DatasetGenerator,
                               pipelines: typing.List[pipe.ExtPipeline], repetitions: int,
                               keep_also_iops: typing.Optional[bool] = False,
                               keep_also_figures: typing.Optional[bool] = False,
                               # export_png: typing.Optional[bool] = False, export_html: typing.Optional[bool] = False,
                               ) -> "ExperimentIoP":
        return ExperimentIoP(pipelines=pipelines, poisoned_datasets=dg.all_datasets, columns=dg.columns,
                             repetitions=repetitions, keep_also_iops=keep_also_iops,
                             keep_also_figures=keep_also_figures) #, export_png=export_png, export_html=export_html)

    def execute_single_pipeline_with_rep(self, X: np.ndarray, y: np.ndarray,
                                         poisoning_idx: np.ndarray, pipeline: pipe.ExtPipeline,
                                         info: base.ExpInfo):

        def _inner(pipeline_: pipe.ExtPipeline):
            result_ = pipeline_.fit_transform(X, y)[0]
            # if the result is 1d, we reshape it to make 2d, otherwise we have issues
            # this way it is treated the same regardless the number of columns in the output.
            result_ = result_ if len(result_.shape) > 1 else result_.reshape(-1, 1)
            return pipeline_, result_

        results: typing.List[typing.Tuple[pipe.ExtPipeline, np.ndarray]] = joblib.Parallel(
            n_jobs=self.repetitions)(joblib.delayed(_inner)(copy.deepcopy(pipeline))
                                     for _ in range(self.repetitions))

        return ExecutePipelineWithRepOutput.from_results(results=[r[0] for r in results], poisoning_idx=poisoning_idx,
                                                         info=info, keep_also_iops=self.keep_also_iops,
                                                         keep_also_figures=self.keep_also_figures)

    def execute_pipeline_on_all_poisoned(self, pipeline: pipe.ExtPipeline) -> ExecutePipelineOutputOnAll:
        results: typing.List[ExecutePipelineWithRepOutput] = joblib.Parallel(n_jobs=len(self.poisoned_datasets))(
            joblib.delayed(self.execute_single_pipeline_with_rep)(
                pipeline=copy.deepcopy(pipeline),
                X=poisoned_dataset.sel(
                    y=[val for val in poisoned_dataset.coords['y'].values
                       if val not in const.DG_IRRELEVANT_COLUMNS]).to_numpy(),
                y=poisoned_dataset.sel(y=const.COORD_LABEL).to_numpy(),
                info=base.ExpInfo(
                    perc_points=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS],
                    perc_features=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES],
                    pipeline_name=pipeline.name
                ),
                poisoning_idx=poisoned_dataset.sel(y=const.COORD_POISONED).to_numpy()
            ) for poisoned_dataset in self.poisoned_datasets.values())

        return ExecutePipelineOutputOnAll.from_results(results=results, keep_also_iops=self.keep_also_iops,
                                                       keep_also_figures=self.keep_also_figures)

    def do(self) -> typing.Dict[str, ExecutePipelineOutputOnAll]:
        def _inner(pipeline_):
            out = self.execute_pipeline_on_all_poisoned(pipeline=pipeline_)
            return pipeline_.name, out

        results: typing.List[typing.Tuple[str, ExecutePipelineOutputOnAll]] = joblib.Parallel(
            n_jobs=len(self.pipelines))(
            joblib.delayed(_inner)(pipeline_=pipeline) for pipeline in self.pipelines)

        return dict(results)
