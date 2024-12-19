import dataclasses
import os
import typing

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate

from . import base
import const
import experiments
import utils


def best_worst(data: pd.DataFrame | pd.Series) -> typing.Dict[str, str]:
    result = {}
    for metric in [const.METRIC_NAME_ACCURACY, const.METRIC_NAME_PRECISION, const.METRIC_NAME_RECALL]:
        sub = data.filter(like=metric)
        worst = sub.idxmin()
        best = sub.idxmax()
        # we keep the prefix only.
        result[f'LOCAL_BEST({metric})'] = worst[:worst.index('_')]
        result[f'LOCAL_WORST({metric})'] = best[:best.index('_')]
    return result

def analyze_individual_delta_self(file_name: str, patterns_to_exclude: typing.Optional[typing.List[str]] = None):
    df = pd.read_csv(file_name)
    # this pd.DataFrame has the following structure:
    # Perc_Points   Perc_Features   Conf1_AVG(DELTA(ACCURACY) Conf1_AVG(DELTA(PRECISION) Conf1_AVG(DELTA(RECALL), etc.
    # 1) retrieve the average delta.
    df = utils.df_negate_filter(df=df, patterns_to_exclude=patterns_to_exclude)
    avg = df.mean()
    avg = avg.drop([const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES])
    # 2.1) retrieve the average for each config.
    avg_global_config = avg.groupby(by=lambda col: col[:col.index('_')]).mean()
    # 2.2) retrieve the worst config according to three metrics
    results = {'GLOBAL': avg_global_config.idxmin()}
    results.update(best_worst(data=avg))
    return results


@dataclasses.dataclass
class ResultPairRaw:
    best_worst: typing.Dict[str, str]
    values: pd.Series


@dataclasses.dataclass
class ResultPairAggregated:
    """
    Attributes:
        best_worst: pd.DataFrame

        values: pd.DataFrame
    """
    best_worst: pd.DataFrame
    values: pd.DataFrame
    # name: str

    @staticmethod
    def decompose(computation_type: base.ComputedAnalysisType,
                  data: typing.List[typing.Dict[base.ComputedAnalysisType, "ResultPairAggregated"]]):
            aggregated_names = []
            aggregated_numbers = []
            for r in data:
                sub = r[computation_type]
                aggregated_names.append(sub.best_worst)
                aggregated_numbers.append(sub.values)
            aggregated_names = pd.concat(aggregated_names)
            aggregated_numbers = pd.concat(aggregated_numbers)
            return aggregated_names, aggregated_numbers

def analyze_individual_delta_(file_name: str, patterns_to_exclude: typing.Optional[typing.List[str]] = None
                              ) -> typing.Dict[base.ComputedAnalysisType, ResultPairRaw]:
    df = pd.read_csv(file_name)
    df = utils.df_negate_filter(df=df, patterns_to_exclude=patterns_to_exclude)

    results_integral = []
    col_to_use = set(df.columns) - {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES}

    for col in col_to_use:
        x = df[const.KEY_PERC_DATA_POINTS]
        y = df[col]
        results_integral.append(integrate.trapezoid(x=x, y=y))

    results_integral = pd.Series(results_integral, index=col_to_use)
    # so basically now we have a pd.Series telling how good the vanilla is for each model and each metric.
    # we now perform the same computations we do in delta_self to retrieve the best/worst local.
    analyzed_integral = best_worst(data=results_integral)

    # do the same computation but on the average.
    results_avg = df.mean()
    results_avg = results_avg.drop([const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES])
    analyzed_avg = best_worst(data=results_avg)

    return {base.ComputedAnalysisType.INTEGRAL: ResultPairRaw(best_worst=analyzed_integral, values=results_integral),#, name=file_name),
            base.ComputedAnalysisType.AVG: ResultPairRaw(best_worst=analyzed_avg, values=results_avg#, name=file_name
                                                          )}

def analyze_all(file_names: typing.List[str], exp_names: typing.List[str],
                patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                # output_file_name: typing.Optional[str] = None
                ) -> typing.Dict[base.ComputedAnalysisType, ResultPairAggregated]:
    results: typing.List[typing.Dict[base.ComputedAnalysisType, ResultPairRaw]] = joblib.Parallel(
        n_jobs=-1)(joblib.delayed(analyze_individual_delta_)(file_name, patterns_to_exclude)
                                         for file_name in file_names)

    raw_results = {c: [] for c in base.ComputedAnalysisType}
    for result in results:
        for computation_type, single_result in result.items():
            raw_results[computation_type].append(single_result)

    # now, we can finally the aggregated pd.DataFrame
    result_pairs = {}
    for computation_type, results_per_computation_type in raw_results.items():

        pair = ResultPairAggregated(best_worst=pd.DataFrame([r.best_worst for r in results_per_computation_type],
                                                            index=exp_names),
                                    values=pd.DataFrame([r.values for r in results_per_computation_type],
                                                        index=exp_names))
        result_pairs[computation_type] = pair

    return result_pairs


def almost_top_aggregate_deltas(sub_dirs: typing.List[str], exp_names: typing.List[str], output_prefix: typing.Optional[str]=None,
                                columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                files_patterns_to_exclude: typing.Optional[typing.List[str]] = None):
    # now basically we will end up with a lot of files if we consider the different test set types and model type,
    # so we do our operation separately *on each test set type/model type*.
    for test_set_type in experiments.TestSetType:
        results = {DELTA_REF: [], DELTA_SELF: []}
        for model_type, delta_type, basic_file_name in [
            (VANILLA, DELTA_SELF, experiments.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF),
            (ORACLE, DELTA_SELF, experiments.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF),
            (f'{VANILLA}_against_{ORACLE}', DELTA_REF, experiments.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED)]:

            file_names_to_use = [
                os.path.join(sub_dir,
                             os.path.join("Output", f'{basic_file_name}_{test_set_type.prefix()}.csv'))
                                      for sub_dir in sub_dirs]

            if files_patterns_to_exclude is not None:
                file_names_to_use = list(filter(lambda f: not base.is_in_patterns(f,
                                                                                  patterns=files_patterns_to_exclude), file_names_to_use))

            result = analyze_all(file_names=file_names_to_use, exp_names=exp_names,
                                  # output_file_name=f'{args.output.replace(".csv", "")}_{test_set_type.prefix()}.csv',
                                  patterns_to_exclude=columns_patterns_to_exclude)
            for k in result.keys():
                result[k].best_worst.rename(lambda ind: f'{model_type}({ind})', axis='index', inplace=True)
                result[k].values.rename(lambda ind: f'{model_type}({ind})', axis='index', inplace=True)
            results[delta_type].append(result)

        # decompose the result.
        for computation_type in base.ComputedAnalysisType:
            for delta_type in [DELTA_REF, DELTA_SELF]:
                aggregated_names, aggregated_numbers = ResultPairAggregated.decompose(computation_type=computation_type,
                                                                                      data=results[delta_type])
                if output_prefix is None:
                    print(aggregated_names)
                    print(aggregated_numbers)
                aggregated_numbers.to_csv(f'{output_prefix}_{delta_type}_{test_set_type.prefix()}_{computation_type.prefix()}_VALUES.csv')
                aggregated_names.to_csv(f'{output_prefix}_{delta_type}_{test_set_type.prefix()}_{computation_type.prefix()}_NAMES.csv')



ORACLE = 'oracle'
VANILLA = 'vanilla'

DELTA_REF, DELTA_SELF = 'DELTA_REF', 'DELTA_SELF'


FILE_TO_PLOT = ['DELTA_REF', 'DELTA_SELF', 'MODEL_QUALITY']


def extract_triple(file_name: str) -> typing.Tuple[experiments.TestSetType, base.DeltaType, base.ModelType]:
    """
    Returns experiments.TestSetType, DELTA_TYPE, MODEL_TYPE.

    This works for results exported from exp_monolithic_models.
    :param file_name:
    :return:
    """
    parts = file_name.split('_')
    # oracle or vanilla is the second one.
    model_type = parts[1]
    if 'quality' == parts[2]:
        delta_type = parts[2]
        next_pos = 3
    else:
        delta_type = f'{parts[2]}_{parts[3]}'
        next_pos = 4
    # and now, finally, the test type.
    # test_set_type_pos = parts[next_pos:]
    test_set_type = None
    test_set_type_got = '_'.join(parts[next_pos:])
    for test_set_type_ in experiments.TestSetType:
        if test_set_type_.prefix() in test_set_type_got:
            test_set_type = test_set_type_
    if test_set_type is None:
        raise ValueError(f'Cannot extract test set type from: {file_name}')
    return test_set_type, base.DeltaType.from_name_in_file(delta_type), base.ModelType.from_name_in_file(model_type)


def extract_model_names(columns: typing.List[str],
                        # columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None
                        ) -> typing.List[str]:
    to_consider = set(columns) - {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES}
    # we first focus on a single metric only.
    metric = const.METRIC_NAME_ACCURACY
    to_consider = list(filter(lambda col: metric in col, to_consider))
    return list(map(lambda col: col.split('_')[0], to_consider))


def almost_top_plot_model_quality(sub_dirs: typing.List[str],
                                  output_prefix: typing.Optional[str]=None,
                                  metrics: typing.List[str] = None,
                                  columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                  files_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                  ):

    # basically we have to collect the quality of all models for the triples below,
    # so that we can compute plots over the same pair of [test set type, delta_type]
    for sub_dir in sub_dirs:
        results_by = {}
        for file in os.listdir(os.path.join(sub_dir, 'Output')):
            complete_file_name = os.path.join(sub_dir, 'Output', file)
            if files_patterns_to_exclude is None or not base.is_in_patterns(complete_file_name, files_patterns_to_exclude):
                triple = extract_triple(file)
                # we do this trick because we want to have oracle and vanilla at the very same place
                # so that we can visually compare them.
                if (triple[0], triple[1]) not in results_by:
                    results_by[(triple[0], triple[1])] = {}
                results_by[(triple[0], triple[1])][triple[2]] = pd.read_csv(complete_file_name)

        # now, we have the results as we want them. It's finally time to plot!
        for (test_set_type, delta_type), dfs in results_by.items():

            # now, when we plot delta_ref, we just have one df under ModelType.VANILLA
            # so we must be careful.
            use_both = False
            if len(dfs) == 2:
                use_both = True
                df_oracle = dfs[base.ModelType.ORACLE]
                df_vanilla = dfs[base.ModelType.VANILLA]

                # we now concatenate the two pd.DataFrame using a join. So we have to first rename
                # columns adding the right prefix.
                df = utils.merge_multiple(dfs=[df_oracle, df_vanilla],
                                                 mask=[(True, f'{base.ModelType.ORACLE.value}_'),
                                                       (True, f'{base.ModelType.VANILLA.value}_')],
                                                 on=[const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES])
            else:
                df = dfs[base.ModelType.VANILLA]

            # vanilla is always there.
            columns = utils.filter_columns(columns=list(dfs[base.ModelType.VANILLA].columns),
                                               patterns_to_exclude=columns_patterns_to_exclude)

            # sort model names. The result is way nicer and easier to figure out.
            model_names = sorted(extract_model_names(columns))
            n_rows = len(model_names)
            n_cols = len(metrics)

            # figsize is of the form (width, height) in inch. A good rule of thumb is 10 for each "quantity", e.g.,
            # 30 x 30 is adequate for three columns and three rows, while 30 x 50 for three columns and five rows.
            # This choice guarantees a proper shape (square-shaped) regardless the number of plots we have.
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10 * n_cols, 10 * n_rows),  # figsize=(30, 30),
                                    layout='constrained')
            fig.suptitle(f'{os.path.basename(sub_dir)}_{delta_type.to_name_in_file()}')

            if use_both:
                # we now have to update the columns because they are changed following the merge.
                columns = [f'{base.ModelType.ORACLE.value}_{col}' if col not in {
                    const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES} else col for col in columns] + \
                          [f'{base.ModelType.VANILLA.value}_{col}' if col not in {
                    const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES} else col for col in columns]

            material = []

            for i, metric in enumerate(metrics):
                for j, model_name in enumerate(model_names):
                    columns_to_keep = [col for col in columns if f'{model_name}_' in col and metric in col]
                    axs_to_use = base.retrieve_axis(axs=axs, second_len=len(model_names), outer_idx=i, inner_idx=j)
                    material.append((axs_to_use, columns_to_keep, f'{metric}({model_name})'))

            # to retrieve the min we concatenate the two dfs.
            y_min = df[list(set(columns) - {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES})].min().min()
            y_max = df[list(set(columns) - {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES})].max().max()

            for axs_to_use, columns_to_keep, title in material:

                base.subplot_in_axs(df=df, axs_to_use=axs_to_use, columns_to_keep=columns_to_keep,
                                    title=title, y_min=y_min, y_max=y_max)

            fig.savefig(f'{output_prefix}_{os.path.basename(sub_dir)}_{delta_type.to_name_in_file()}_{test_set_type.prefix()}.png')
            fig.clf()

def read_sub_dirs_and_exp_names(input_directory: str) -> typing.Tuple[typing.List[str], typing.List[str]]:
    # now, we have to crawl in the directory and locate the files.
    sub_dirs = os.listdir(input_directory)
    sub_dirs = list(filter(lambda f: os.path.isdir(os.path.join(input_directory, f)), sub_dirs))
    exp_names = sub_dirs
    sub_dirs = list(map(lambda f: os.path.join(input_directory, f), sub_dirs))
    return sub_dirs, exp_names


def top_delta_aggregated(args):
    sub_dirs, exp_names = read_sub_dirs_and_exp_names(args.input_directory)
    almost_top_aggregate_deltas(sub_dirs=sub_dirs, output_prefix=args.output_prefix, exp_names=exp_names,
                                columns_patterns_to_exclude=args.columns_patterns_to_exclude,
                                files_patterns_to_exclude=args.files_patterns_to_exclude)


def top_plots(args):
    sub_dirs, _ = read_sub_dirs_and_exp_names(args.input_directory)
    almost_top_plot_model_quality(sub_dirs=sub_dirs, output_prefix=args.output_prefix,
                                  metrics=args.metrics,
                                  columns_patterns_to_exclude=args.columns_patterns_to_exclude,
                                  files_patterns_to_exclude=args.files_patterns_to_exclude)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-directory', '-i', type=str, required=True,
                        help='Top-level directory with input files')
    parser.add_argument('--output-prefix', type=str, default=None, help='Output file')
    parser.add_argument('--columns-patterns-to-exclude', type=str, nargs='+', required=False)
    parser.add_argument('--files-patterns-to-exclude', type=str, nargs='+', required=False)

    sub_parsers = parser.add_subparsers()

    parser_delta_aggregated = sub_parsers.add_parser('build-aggregated-deltas')
    parser_delta_aggregated.set_defaults(func=top_delta_aggregated)

    parser_plots = sub_parsers.add_parser('build-plots')
    parser_plots.add_argument('--metrics', type=str, nargs='+', required=True,
                                           help='Metrics to include in the plot')
    parser_plots.set_defaults(func=top_plots)

    args__ = parser.parse_args()
    args__.func(args__)


# python -m post.base_models --input-directory \
# /Users/nicola/UniWork/Research/PALM/WP01/EnsembleResultsNew/20240430_NewTargeted/BaseModels --output out.csv \
# --columns-patterns-to-exclude svm

