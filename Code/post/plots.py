import enum
import os.path
import re
import typing
import warnings

import pandas as pd
from matplotlib import pyplot as plt

from . import base
import const
import utils

class PlotMergedMode(enum.Enum):
    SAME_N_DIFFERENT_PIPE = 'SAME_N_DIFFERENT_PIPE'
    DIFFERENT_N_SAME_PIPE = 'DIFFERENT_N_SAME_PIPE'

    @staticmethod
    def from_str(val: str) -> "PlotMergedMode":
        if val == PlotMergedMode.SAME_N_DIFFERENT_PIPE.value:
            return PlotMergedMode.SAME_N_DIFFERENT_PIPE
        return PlotMergedMode.DIFFERENT_N_SAME_PIPE


# import utils


def almost_top_compute_arbitrary_delta_ref(
        df: pd.DataFrame, metrics: typing.List[str], output_dir_delta: str,
        patterns: typing.List[base.PipelineNamesToMatch], pre_prefix: str,
        # output_dir_plot: typing.Optional[str] = None,
        columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None) -> pd.DataFrame:
    """
    This function is meant to perform additional "delta_ref" computation and plot beyond those that are done
    in the main experiment.

    Let us recall what delta_ref does. Let res(a, m, p) be the result retrieved from
    - model/pipeline a (e.g., TSUSC)
    - metric m (e.g., accuracy)
    - percentage of poisoning p (e.g., eps=10).

    Delta_ref is defined as res(a, m, p) - res(BASE, m, p), where BASE is a baseline model (e.g., TSUSC, monolithic).
    Thus, *positive values of delta_ref* means that a is better than BASE.

    It takes as input
    - the "super merged file", i.e., `Merged/model_ensemble.csv`
    - the models to compare against.

    The function allows to compare one model against several baselines. In other words, if we have one model
    (e.g., IoP-based) and two baselines (e.g., TSUSC, random), there will be one plot (actually one for each metric
    and N) with two lines showing the two delta_ref of the first model against the second, and the first against the
    third.

    The function also produces a "big" csv including all the retrieved delta_ref.

    The format of these comparison patterns is thus a list of:

    - "our" model against
        - list of
            - baseline model, which is identified by its pipeline name as present in the csv file (e.g., TSUSC will
              be  matched against TSUSC N3, TSUSC N9, etc.)
            - prefix_to_use_for_export: the name of the column to use in the big, merged csv containing the delta_ref
              of "our" against "baseline". For instance, let us assume we are comparing a round-robin risk against
              TSUSC, this attribute can be called `rr_vs_TSUSC`.

    :param df:
    :param metrics:
    :param output_dir_delta:
    :param patterns:
    :param pre_prefix:
    :param columns_patterns_to_exclude:
    :return:
    """
    # df = df[utils.negate_filter(columns=df.columns, patterns_to_exclude=columns_patterns_to_exclude)]
    df = utils.df_negate_filter(df=df, patterns_to_exclude=columns_patterns_to_exclude)

    results = []
    for pattern in patterns:
        # let's start with our model
        df_other = df.filter(regex=f'^{re.escape(pattern.other)}')
        # df_other = df.filter(like=pattern.other)
        # print(f'For other "{pattern.other}" I have: {df_other.columns.to_list()}, len: {len(df_other)}')
        if len(df_other.columns) == 0:
            raise ValueError(f'No values found for pattern.other: {pattern.other}')

        # we extract the different Ns
        Ns, _ = base.extract_Ns(df_other.columns)

        # this contains the results, we create one target that is continuously merged.
        target = pd.DataFrame()

        # and we compare it against the baselines.
        for against in pattern.against:

            # let's extract the baseline according to the specified columns.
            # df_against = df.filter(like=against.col_name)
            # we need escape since it may contain '+' and so on which are special for regex.
            df_against = df.filter(regex=f'^{re.escape(against.col_name)}')
            if len(df_against.columns) == 0:
                raise ValueError(f'Patterns against \'{against.col_name}\' not found in columns {list(df.columns)}')

            # now, we have to "group" by N, and then make sure columns
            # are only called as the metrics, otherwise we cannot performa a correct subtraction.

            for n in Ns:
                # now, we perform the comparison N by N.
                sub_against_col = base.extract_columns_by_metrics_and_n(columns=df_against.columns, metrics=metrics,
                                                                        # in case we request delta-ref against the monolithic
                                                                        n=n if 'MONO' not in against.col_name else None,
                                                                        Ns=Ns)
                sub_other_col = base.extract_columns_by_metrics_and_n(columns=df_other.columns, metrics=metrics, n=n,
                                                                      Ns=Ns)
                # make sure the columns are of the same length, otherwise we cannot perform a correct subtraction.
                if len(sub_against_col) != len(sub_other_col):
                    raise ValueError(f'Patterns do not match exactly (N={n}). For against "{against.col_name}", '
                                     f'I got: {sub_against_col} from {list(df_against.columns)}. '
                                     f'For other "{pattern.other}" I got: {sub_other_col}')

                if len(sub_against_col) == 0:
                    warnings.warn(f'No columns found for {against.col_name} and N={n}')

                sub_against_df = df_against[sub_against_col]
                sub_other_df = df_other[sub_other_col]

                # What we want to have are two pd.DataFrame of the very same shape and columns, for instance:
                # perc_points perc_features Accuracy Recall Precision
                #   1               0           90      90  90
                # ...
                # However, the two sub-frames have different columns for Accuracy and Recall because they refer
                # to different pipelines, e.g., `TSUSC N9_AVG(Accuracy)` for the baseline and `Risk+RR N9_AVG(Accuracy)`
                # for our. So, we rename the columns to temporarily keep only the metrics, as shown above.
                def renamer(col_):
                    for metric in metrics:
                        if metric in col_:
                            return metric
                    return col_

                sub_against_df = sub_against_df.rename(renamer, axis='columns')
                sub_other_df = sub_other_df.rename(renamer, axis='columns')

                # ok now the two the DataFrame have equals columns and can be actually subtracted.
                result = sub_other_df - sub_against_df
                # now, we perform the reversed action, re-adding to each column the specified prefix (e.g.,
                # `RR_vs_TSUSC`), the value of N (e.g., `N9`), while keeping the column format of `AVG(metric)`
                # (e.g., in the end we have `RR_vs_TSUSC N9_AVG(Accuracy)`).
                result = result.rename(
                    lambda col: f'{against.prefix_to_use_for_export} N{n:0{base.get_N_digits(Ns)}d}_AVG({col})',
                    axis='columns')
                # now, add perc points and perc features in the first and second position
                result.insert(0, const.KEY_PERC_DATA_POINTS, df[const.KEY_PERC_DATA_POINTS])
                result.insert(1, const.KEY_PERC_FEATURES, df[const.KEY_PERC_FEATURES])
                kwargs = base.get_recursive_merge_kwargs(target=target)
                # print(f'Merging on {kwargs}')
                target = pd.merge(target, result, **kwargs)

        results.append(target)

    final_df = results[0]
    for i, result in enumerate(results[1:], 1):
        try:
            final_df = pd.merge(final_df, result, on=list(base.POINTS_PERC_FEATURES_), how='inner')
        except KeyError as e:
            print(f'Error at {patterns[i]}')
            raise e
    # we're almost done. We create one final csv where we introduce a counter (the number of times is > 0).
    # First, we remove perc_points and features, we are not interested in them.
    sub_final_df = final_df[[col for col in final_df.columns if col not in base.POINTS_PERC_FEATURES_]]
    count = sub_final_df[sub_final_df > 0].count() / sub_final_df.count()
    pd.DataFrame({'Count': count, 'Avg(DeltaRef)': sub_final_df.mean()}).to_csv(
        # the index is the "pipeline" name, so better keep it.
        os.path.join(output_dir_delta, f'{pre_prefix}_delta_ref_summary.csv'), index=True)
    final_df.to_csv(os.path.join(output_dir_delta, f'{pre_prefix}_delta_ref.csv'), index=False)
    return final_df


def almost_top_plot_arbitrary_delta_ref(df: pd.DataFrame, metrics: typing.List[str],
                                        output_dir: str, patterns: typing.List[base.PipelineNamesToMatch],
                                        stat_summary_df: typing.Optional[pd.DataFrame] = None,
                                        pre_prefix: typing.Optional[str] = None,
                                        columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None
                                        ):
    # df already contains all the patterns.
    for pattern in patterns:
        target = pd.DataFrame()
        for against in pattern.against:
            df_against = df.filter(like=against.prefix_to_use_for_export).copy()
            df_against = base.add_basic_info_to_df(sub_df=df_against, full_df=df)
            target = pd.merge(target, df_against, **base.get_recursive_merge_kwargs(target=target))

        prefix_to_use = pre_prefix if pre_prefix is not None else ''
        prefix_to_use = f'{prefix_to_use}{pattern.plot_prefix}'

        plot_varying_quality_metric_and_N(df=target, metrics=metrics, output_dir=output_dir,
                                          prefix=prefix_to_use, mode=PlotMergedMode.SAME_N_DIFFERENT_PIPE,
                                          stat_summary_df=stat_summary_df,
                                          columns_patterns_to_exclude=columns_patterns_to_exclude)


def plot_varying_quality_metric_and_N(df: pd.DataFrame,
                                      metrics: typing.List[str],
                                      output_dir: str,
                                      prefix: str,
                                      mode: PlotMergedMode,  # = PlotMergedMode.PLOT_MERGED_MODE_SAME_N_DIFFERENT_PIPE,
                                      columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                      columns_patterns_to_include: typing.Optional[typing.List[str]] = None,
                                      always_keep_monolithic: bool = False,
                                      stat_summary_df: typing.Optional[pd.DataFrame] = None,
                                      ):
    """

    :param columns_patterns_to_include:
    :param stat_summary_df:
    :param df:
    :param metrics:
    :param output_dir:
    :param prefix:
    :param mode:
    :param columns_patterns_to_exclude:
    :param always_keep_monolithic: if True, the lines corresponding to the monolithic model (when applicable) are always plot.
    :return:
    """

    columns = utils.filter_columns(columns=df.columns, patterns_to_exclude=columns_patterns_to_exclude,
                                   patterns_to_include=columns_patterns_to_include)
    Ns, all_but_the_first = base.extract_Ns(columns)
    # pipeline_names includes *also* monolithic model-related pipelines.
    pipeline_names = base.extract_pipelines(columns)

    if set(columns) == base.POINTS_PERC_FEATURES_:
        # no need to go on. This config has been entirely filtered (which may be fine).
        return

    # sort so that we have N in ascending order.
    Ns.sort()

    y_min, y_max = base.get_ymin_and_ymax(df[all_but_the_first])

    if mode == PlotMergedMode.SAME_N_DIFFERENT_PIPE:
        n_rows, n_cols = len(Ns), len(metrics)
    else:
        n_rows, n_cols = len(pipeline_names), len(metrics)

    # figsize is of the form (width, height) in inch. A good rule of thumb is 10 for each "quantity", e.g.,
    # 30 x 30 is adequate for three columns and three rows, while 30 x 50 for three columns and five rows.
    # This choice guarantees a proper shape (square-shaped) regardless the number of plots we have.
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10 * n_cols, 10 * n_rows),  # figsize=(30, 30),
                            layout='constrained')
    fig.suptitle(prefix)

    material = []
    if mode == PlotMergedMode.SAME_N_DIFFERENT_PIPE:
        # this is the case where the parameter always_keep_monolithic is applicable.
        for i, metric in enumerate(metrics):
            for j, n in enumerate(Ns):
                # retrieve the columns to keep.
                if n == 0:
                    # to_keep = [col for col in columns if f'MONO_AVG({metric})' in col]
                    to_keep = [col for col in columns if f'MONO' in col and metric in col]
                else:
                    to_keep = [col for col in columns if
                               (f'N{n}_' in col or f'N{n:0{base.get_N_digits(Ns)}d}' in col) and metric in col]
                    if always_keep_monolithic:
                        # add the monolithic lines.
                        to_keep += [col for col in columns if 'MONO' in col and metric in col]

                axs_to_use = base.retrieve_axis(axs=axs, second_len=len(Ns), outer_idx=i, inner_idx=j)
                material.append((axs_to_use, to_keep, metric, n))
    else:
        # same pipe but different N.
        for i, metric in enumerate(metrics):
            for j, pipeline_name in enumerate(pipeline_names):
                # now, retrieve all matching pipelines.
                to_keep = [col for col in columns if f'{pipeline_name} N' in col and metric in col]
                if always_keep_monolithic:
                    # add the monolithic lines.
                    to_keep += [col for col in columns if 'MONO' in col and metric in col]
                axs_to_use = base.retrieve_axis(axs=axs, second_len=len(pipeline_names), outer_idx=i, inner_idx=j)
                material.append((axs_to_use, to_keep, metric, pipeline_name))

    # and now that we have all the information we can plot.
    for axs_to_use, columns_to_keep, metric, n in material:
        print(f'Columns: {columns_to_keep} with {metric}')
        base.subplot_in_axs(df=df, axs_to_use=axs_to_use, columns_to_keep=columns_to_keep, title=f'{metric} {n}',
                            y_min=y_min, y_max=y_max, stat_summary_df=stat_summary_df)

    fig.savefig(os.path.join(output_dir, f'{prefix}.png'), dpi=150)
    fig.clf()


def almost_top_merge_model_quality(old_and_ref: pd.DataFrame, new: pd.DataFrame, ) -> pd.DataFrame:
    # we take the new dataframe and remove old columns related to the monolithic models.
    columns_to_keep = [col for col in new.columns if 'MONO' not in col]
    new = new[columns_to_keep]

    common = set(old_and_ref.columns).intersection(set(new.columns))
    if common != base.POINTS_PERC_FEATURES_:
        raise ValueError(f'There are columns in common but they should not: {common}')

    # now we can do the merge.
    merged = pd.merge(left=old_and_ref, right=new, on=list(base.POINTS_PERC_FEATURES_), validate='one_to_one')
    return merged


def top_merge_model_quality(args):
    old_and_ref = pd.read_csv(args.input_file_old_and_ref)
    new = pd.read_csv(args.input_file_new)
    result = almost_top_merge_model_quality(old_and_ref=old_and_ref, new=new)
    result.to_csv(args.output_file, index=False)


def almost_top_plot_model_quality_simple(source: pd.DataFrame, prefix: str,
                                         output_dir: str):
    # create a sub DataFrame excluding STD.
    columns_no_std = utils.filter_columns(columns=source.columns, patterns_to_exclude=['STD'])

    y_min, y_max = base.get_ymin_and_ymax(source[columns_no_std])

    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10 * ncols, 10 * nrows), layout='constrained')
    fig.suptitle(prefix)

    for i, metric in enumerate(columns_no_std):
        axs[0, 1].set_ylim([y_min, y_max])
        source.plot(x=const.KEY_PERC_DATA_POINTS, y=metric, ax=axs[0, i], title=f'{metric}',
                    yerr=source[metric.replace(const.PREFIX_AVG, const.PREFIX_STD)],
                    fontsize=20)
    fig.savefig(os.path.join(output_dir, f'{prefix}.png'), dpi=300)


def almost_top_plot_monolithic_quality_in_merged(source: pd.DataFrame, prefix: str,
                                                 output_dir: str, metrics: typing.List[str]):
    columns = [col for col in source.columns if const.MONOLITHIC_VANILLA_PIPELINE_NAME in col or
               const.MONOLITHIC_ORACLED_PIPELINE_NAME in col]

    y_min, y_max = base.get_ymin_and_ymax(source[columns])

    columns = columns + const.INFO_KEY_LIST

    nrows, ncols = 1, 3
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10 * ncols, 10 * nrows), layout='constrained')
    fig.suptitle(prefix)

    materials = []
    for i, metric in enumerate(metrics):
        to_keep = [col for col in columns if f'AVG({metric})' in col]
        materials.append((axs[i], to_keep, metric))

    for axs, columns_to_keep, metric in materials:
        print(f'Columns: {columns_to_keep} with {metric}')
        base.subplot_in_axs(df=source, axs_to_use=axs, title=f'{metric}', y_min=y_min, y_max=y_max,
                            columns_to_keep=columns_to_keep)

    fig.savefig(os.path.join(output_dir, f'{prefix}.png'), dpi=300)


def top_compute_arbitrary_delta_ref(args):
    patterns = base.read_patterns(patterns_file=args.patterns_file, patterns=args.patterns)

    almost_top_compute_arbitrary_delta_ref(df=pd.read_csv(args.input_file),
                                           # output_dir_plot=args.output_dir_plot,
                                           output_dir_delta=args.output_dir,
                                           patterns=patterns, metrics=args.metrics, pre_prefix=args.pre_prefix,
                                           columns_patterns_to_exclude=args.columns_patterns_to_exclude)


def almost_top_compute_summary(source: pd.DataFrame,
                               metrics: typing.Optional[typing.List[str]] = None,
                               columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                               columns_patterns_to_include: typing.Optional[typing.List[str]] = None,
                               perc_points_min_value: typing.Optional[float] = None,
                                perc_points_max_value: typing.Optional[float] = None,
                               ) -> pd.DataFrame:
    cols = base.filter_columns_and_metrics(df=source, columns_patterns_to_exclude=columns_patterns_to_exclude,
                                           columns_patterns_to_include=columns_patterns_to_include,
                                           metrics=metrics)
    source = source[cols]

    if perc_points_min_value is not None:
        source = source[source[const.KEY_PERC_DATA_POINTS] >= perc_points_min_value]
    if perc_points_max_value is not None:
        source = source[source[const.KEY_PERC_DATA_POINTS] <= perc_points_max_value]

    summary = base.compute_summary(source)
    merged = base.merge_summary(summary)
    # just rename the column since they use an enum
    merged.rename(lambda col: col.value, inplace=True, axis='columns')
    return merged


def almost_top_compact_delta_ref(source: pd.DataFrame,
                                 metrics: typing.List[str],
                                 columns_patterns_to_exclude: typing.Optional[
                                     typing.List[str]] = None, ) -> pd.DataFrame:
    # we work on the transposed delta-ref, where everything is in the index.
    idx = utils.filter_columns(columns=source.index, patterns_to_exclude=columns_patterns_to_exclude)
    cols = [col for col in source.columns if base.is_in_patterns(col, metrics)]

    def expand(row: pd.Series) -> pd.Series:
        result_ = []
        for m in metrics:
            single_ = pd.Series([
                row[f'{const.PREFIX_AVG}({m})({base.ComputedAnalysisType.INTEGRAL.value.lower()})'] > 0,
                row[f'{const.PREFIX_AVG}({m})({base.ComputedAnalysisType.AVG.value.lower()})']],
                index=[f'{const.PREFIX_AVG}({m})({base.ComputedAnalysisType.INTEGRAL.value.lower()})',
                       f'{const.PREFIX_AVG}({m})({base.ComputedAnalysisType.AVG.value.lower()})'])
            result_.append(single_)
        return pd.concat(result_, )

    # apply the filter to the pd.DataFrame
    source = source[cols]
    source = source.loc[idx]
    result: pd.DataFrame = source.apply(lambda row: expand(row), axis='columns', result_type='expand')

    # finally, we sort.
    def key_fn(idxs_):
        return [idx_[idx_.index('vs_') + 1:] if 'vs_' in idx_ else idx_ for idx_ in idxs_]

    result.sort_index(inplace=True, key=key_fn)
    return result


# df=base.read_stat_summary(path_or_df=args.input_file)

def top_compute_summary(args):
    result = almost_top_compute_summary(source=pd.read_csv(args.input_file),
                                        columns_patterns_to_exclude=args.columns_patterns_to_exclude,
                                        columns_patterns_to_include=args.columns_patterns_to_include,
                                        metrics=args.metrics, perc_points_min_value=args.perc_points_min_value,
                                        perc_points_max_value=args.perc_points_max_value,)
    result.to_csv(args.output_file)


def top_plot_varying_quality_metric_and_N(args):
    stat_ = base.get_stat_summary_df_from_args(args)

    if len(args.input_file) > 1:
        if args.input_file_prefix is None or len(args.input_file_prefix) != len(args.input_file):
            raise ValueError(f'Got {len(args.input_file)} input files, but {args.input_file_prefix} prefixes')
        # now merge them
        source = utils.merge_multiple([pd.read_csv(input_file) for input_file in args.input_file],
                                      on=list(base.POINTS_PERC_FEATURES_),
                                      mask=[(True, prefix) for prefix in args.input_file_prefix], )
        if stat_ is not None:
            raise ValueError('stat_ not supported when using multiple inputs')
    else:
        source = pd.read_csv(args.input_file[0])

    plot_varying_quality_metric_and_N(df=source,  # df=pd.read_csv(args.input_file),
                                      metrics=args.metrics,
                                      output_dir=args.output_dir,
                                      prefix=args.prefix,
                                      mode=PlotMergedMode.from_str(args.mode),
                                      columns_patterns_to_exclude=args_.columns_patterns_to_exclude,
                                      columns_patterns_to_include=args.columns_patterns_to_include,
                                      always_keep_monolithic=args.always_keep_monolithic,
                                      stat_summary_df=stat_, )


def top_plot_arbitrary_delta_ref(args):
    patterns = base.read_patterns(patterns_file=args.patterns_file, patterns=args.patterns)
    almost_top_plot_arbitrary_delta_ref(df=pd.read_csv(args.input_file),
                                        metrics=args.metrics,
                                        output_dir=args.output_dir,
                                        patterns=patterns, stat_summary_df=base.get_stat_summary_df_from_args(args),
                                        pre_prefix=args.pre_prefix,
                                        columns_patterns_to_exclude=args.columns_patterns_to_exclude, )


def top_transpose_arbitrary_delta_ref(args):
    df = base.transpose_arbitrary_delta_ref(df=base.read_stat_summary(path_or_df=args.input_file), )
    df.to_csv(args.output_file)

    # now, we compute the compact version.
    if args.compute_compact:
        compact = almost_top_compact_delta_ref(source=df, metrics=args.metrics,
                                               columns_patterns_to_exclude=args.columns_patterns_to_exclude, )
        compact.to_csv(args.output_file.replace('.csv', '_compact.csv'))


def top_plot_model_quality_simple(args):
    source = pd.read_csv(args.input_file)
    almost_top_plot_model_quality_simple(source=source, output_dir=args.output_dir, prefix=args.prefix)


def top_plot_monolithic_quality_in_merged(args):
    almost_top_plot_monolithic_quality_in_merged(source=pd.read_csv(args.input_file),
                                                 output_dir=args.output_dir, prefix=args.prefix,
                                                 metrics=args.metrics, )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers()

    parser_plot_merged = sub_parsers.add_parser('plot-merged')
    parser_plot_merged.add_argument('--metrics', type=str, nargs='+', required=True,
                                    help='Metrics to include in the plot')
    parser_plot_merged.add_argument('--columns-patterns-to-exclude', type=str, nargs='+', required=False,
                                    help='Columns whose content matches the given one are excluded')
    parser_plot_merged.add_argument('--columns-patterns-to-include', type=str, nargs='+', required=False,
                                    help='Columns whose content matches the given one are included')
    parser_plot_merged.add_argument('--input-file', type=str, required=True, nargs='+',
                                    help='Path to the input file(s)')
    parser_plot_merged.add_argument('--input-file-prefix', type=str, required=False, nargs='+',
                                    help='Postfix to use while merging input files')
    parser_plot_merged.add_argument('--output-dir', type=str, required=True,
                                    help='Path to the output directory')
    parser_plot_merged.add_argument('--prefix', type=str, required=True,
                                    help='Prefix of the output file')
    parser_plot_merged.add_argument('--mode', type=str, required=False,
                                    choices=[PlotMergedMode.DIFFERENT_N_SAME_PIPE.value,
                                             PlotMergedMode.SAME_N_DIFFERENT_PIPE.value])
    parser_plot_merged.add_argument('--always-keep-monolithic', type=bool, required=False, default=False,
                                    help='Include or not the lines corresponding to the monolithic model alongside '
                                         'other pipelines. It is applicable only when the monolithic model has '
                                         'distinct lines to plot.')
    parser_plot_merged.add_argument('--stat-summary-file', type=str, required=False, default=None,
                                    help='Path to the summary stats file (if any), created using the '
                                         'command summary-stat')
    # parser_plot_merged.add_argument('--output-playable', type=bool, required=False, default=False, )
    parser_plot_merged.set_defaults(func=top_plot_varying_quality_metric_and_N)

    parser_arbitrary_delta_ref_compute = sub_parsers.add_parser('compute-arbitrary-delta-ref')
    parser_arbitrary_delta_ref_compute.add_argument('--metrics', type=str, nargs='+', required=True,
                                                    help='Metrics to include in the plot')
    parser_arbitrary_delta_ref_compute.add_argument('--columns-patterns-to-exclude', type=str, nargs='+',
                                                    required=False,
                                                    help='Columns whose content matches the given one are excluded')
    parser_arbitrary_delta_ref_compute.add_argument('--input-file', type=str, required=True,
                                                    help='Path to the input file')
    # parser_arbitrary_delta_ref.add_argument('--output-dir-plot', type=str, required=False, )
    parser_arbitrary_delta_ref_compute.add_argument('--output-dir', type=str, required=True, )
    parser_arbitrary_delta_ref_compute.add_argument('--pre-prefix', type=str, required=True,
                                                    help='Prefixes of the output file')
    parser_arbitrary_delta_ref_compute.add_argument('--patterns', type=str, required=False, )
    parser_arbitrary_delta_ref_compute.add_argument('--patterns-file', type=str, required=False, )
    parser_arbitrary_delta_ref_compute.add_argument('--stat-summary-file', type=str, required=False, default=None,
                                                    help='Path to the summary stats file (if any), created using the '
                                                         'command summary-stat')
    parser_arbitrary_delta_ref_compute.set_defaults(func=top_compute_arbitrary_delta_ref, )

    parser_arbitrary_delta_ref_plot = sub_parsers.add_parser('plot-arbitrary-delta-ref')
    parser_arbitrary_delta_ref_plot.add_argument('--metrics', type=str, nargs='+', required=True, )
    parser_arbitrary_delta_ref_plot.add_argument('--input-file', type=str, required=True,
                                                 help='Path to the input file')
    parser_arbitrary_delta_ref_plot.add_argument('--output-dir', type=str, required=True, )
    parser_arbitrary_delta_ref_plot.add_argument('--pre-prefix', type=str, required=True,
                                                 help='Prefixes of the output file')
    parser_arbitrary_delta_ref_plot.add_argument('--columns-patterns-to-exclude', type=str, nargs='+',
                                                 required=False,
                                                 help='Columns whose content matches the given one are excluded')
    parser_arbitrary_delta_ref_plot.add_argument('--patterns', type=str, required=False, )
    parser_arbitrary_delta_ref_plot.add_argument('--patterns-file', type=str, required=False, )
    parser_arbitrary_delta_ref_plot.add_argument('--stat-summary-file', type=str, required=False, default=None,
                                                 help='Path to the summary stats file (if any), created using the '
                                                      'command summary-stat')
    parser_arbitrary_delta_ref_plot.set_defaults(func=top_plot_arbitrary_delta_ref, )

    parser_summary_stat = sub_parsers.add_parser('summary-stat')
    parser_summary_stat.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    parser_summary_stat.add_argument('--output-file', type=str, required=True, help='Path to the output file')
    parser_summary_stat.add_argument('--columns-patterns-to-exclude', type=str, nargs='+',
                                     required=False,
                                     help='Columns whose content matches the given one are excluded')
    parser_summary_stat.add_argument('--metrics', type=str, nargs='+', required=False,
                                     help='Metrics to include',
                                     default=[const.METRIC_NAME_ACCURACY, const.METRIC_NAME_RECALL,
                                              const.METRIC_NAME_PRECISION])
    parser_summary_stat.add_argument('--columns-patterns-to-include', type=str, nargs='+', required=False,
                                     help='Columns whose content matches the given one are included')
    parser_summary_stat.add_argument('--perc-points-min-value', type=float, required=False, default=0.0)
    parser_summary_stat.add_argument('--perc-points-max-value', type=float, required=False, default=100.0)
    parser_summary_stat.set_defaults(func=top_compute_summary, )

    parser_arbitrary_delta_ref_transpose = sub_parsers.add_parser('transpose-arbitrary-delta-ref')
    parser_arbitrary_delta_ref_transpose.add_argument('--input-file', type=str, required=True,
                                                      help='Path to the input file')
    parser_arbitrary_delta_ref_transpose.add_argument('--output-file', type=str, required=True,
                                                      help='Path to the output file')
    parser_arbitrary_delta_ref_transpose.add_argument('--columns-patterns-to-exclude', type=str, nargs='+',
                                                      required=False,
                                                      help='Columns whose content matches the given one are excluded')
    parser_arbitrary_delta_ref_transpose.add_argument('--metrics', type=str, nargs='+', required=False,
                                                      help='Metrics to include in the compact',
                                                      default=[const.METRIC_NAME_ACCURACY, const.METRIC_NAME_RECALL,
                                                               const.METRIC_NAME_PRECISION])
    parser_arbitrary_delta_ref_transpose.add_argument('--compute-compact', type=bool, required=False,
                                                      default=True)
    parser_arbitrary_delta_ref_transpose.set_defaults(func=top_transpose_arbitrary_delta_ref, )

    parser_plot_model_quality_simple = sub_parsers.add_parser('plot-model-quality-simple')
    parser_plot_model_quality_simple.add_argument('--input-file', type=str, required=True, )
    parser_plot_model_quality_simple.add_argument('--output-dir', type=str, required=True, )
    parser_plot_model_quality_simple.add_argument('--prefix', type=str, required=True)
    parser_plot_model_quality_simple.set_defaults(func=top_plot_model_quality_simple, )

    parser_plot_monolithic_quality_in_merged = sub_parsers.add_parser('plot-merged-mono')
    parser_plot_monolithic_quality_in_merged.add_argument('--input-file', type=str, required=True, )
    parser_plot_monolithic_quality_in_merged.add_argument('--output-dir', type=str, required=True, )
    parser_plot_monolithic_quality_in_merged.add_argument('--prefix', type=str, required=True)
    parser_plot_monolithic_quality_in_merged.add_argument('--metrics', type=str, nargs='+', required=True,
                                                          help='Metrics to include in the plot')
    parser_plot_monolithic_quality_in_merged.set_defaults(func=top_plot_monolithic_quality_in_merged, )

    parser_merge_model_quality = sub_parsers.add_parser('merge-model-quality')
    parser_merge_model_quality.add_argument('--input-file-old-and-ref', type=str, required=True, )
    parser_merge_model_quality.add_argument('--input-file-new', type=str, required=True, )
    parser_merge_model_quality.add_argument('--output-file', type=str, required=True, )
    parser_merge_model_quality.set_defaults(func=top_merge_model_quality, )

    args_ = parser.parse_args()
    args_.func(args_)
