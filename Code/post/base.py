import argparse
import dataclasses
import enum
import textwrap
import typing
import warnings

import cycler
import json5 as json
import mashumaro
# from mashumaro.codecs import json
import pandas as pd
from scipy import integrate

import const
import utils

POINTS_PERC_FEATURES_ = {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES}

EPSILON = 0.1


def retrieve_axis(axs, second_len, outer_idx, inner_idx):
    if len(axs.shape) > 1:
        axs_to_use = axs[inner_idx, outer_idx]
    else:
        # when the axis is 1d, we cannot use [j, i]
        # ATTENTION it was 0, may cause issues (with 1, it works for monolithic models plots)
        if second_len > 1:
            axs_to_use = axs[inner_idx]
        else:
            axs_to_use = axs[outer_idx]
    return axs_to_use


def subplot_in_axs(*, df: pd.DataFrame, axs_to_use, y_min: float, columns_to_keep: typing.List[str], y_max: float,
                   title: str, stat_summary_df: typing.Optional[pd.DataFrame] = None, ):
    axs_to_use.set_ylim([y_min, y_max])
    # so that when we reuse the color at least the line changes.
    # check out: https://matplotlib.org/stable/users/explain/artists/color_cycle.html#sphx-glr-users-explain-artists-color-cycle-py
    # The '*' operation creates a zip-like multiplication so that we never run out of colors and line styles.
    axs_to_use.set_prop_cycle(cycler.cycler(linestyle=['solid', 'dashed', 'dotted', 'dashdot']) *
                              cycler.cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                                   'tab:brown', 'tab:pink',
                                                   'tab:gray', 'tab:olive', 'tab:cyan', 'yellow', 'black']))
    # check out: https://matplotlib.org/stable/gallery/color/named_colors.html for more colors.
    labels = columns_to_keep
    if stat_summary_df is not None:
        labels = [build_str_label_from_stats(col, stat_summary_df) for col in labels]
    # labels = ['\n'.join(textwrap.wrap(col, 80)) for col in columns_to_keep]
    labels = ['\n'.join(textwrap.wrap(col, 95)) for col in labels]
    df.plot(x=const.KEY_PERC_DATA_POINTS, y=columns_to_keep, ax=axs_to_use, title=title,
            fontsize=20)
    axs_to_use.hlines(y=0, color='r', xmin=df[const.KEY_PERC_DATA_POINTS].min(),
                      xmax=df[const.KEY_PERC_DATA_POINTS].max(),
                      linestyle='dashed')
    axs_to_use.legend(labels, loc='upper center',  # bbox_to_anchor=(0.5, 1.55),
                      prop={'size': 10})


def split_string(a: str) -> str:
    # https://stackoverflow.com/questions/18854620/whats-the-best-way-to-split-a-string-into-fixed-length-chunks-and-work-with-the
    length = 20
    chunks = [a[0 + i:length + i] for i in range(0, len(a), length)]
    return '\n'.join(chunks)


def extract_n_from_chunk_in_column(chunk: str) -> int:
    last = chunk.replace('N', '')
    last = last[:last.index('_')]
    return int(last)


def extract_n_from_pipeline_name(name: str) -> int:
    last = name.split(' ')[-1]
    return int(last.replace('N', ''))

def map_fn_N_only(col_name: str) -> int:
    # col_name = col_name.replace('\n', '')

    # here, we split by ' '.
    col = col_name.split()
    # then, we retrieve the last part that contains N. Note that
    # the column can also refer to the monolithic model. In this case,
    # it will be called like 'BASE_AVG(Recall)'.
    if len(col) > 1:
        last: str = col[-1]
        # we remove everything from '_' on and 'N'.
        # last = last.replace('N', '')
        # last = last[:last.index('_')]
        result = extract_n_from_chunk_in_column(last)
    else:
        # it may refer to the monolithic model:
        if 'MONO' in col_name and ' N' not in col_name:
            # monolithic model
            result = 0
        else:
            # it is not the monolithic model, but rather the result of some prior manipulation.
            # so we try some additional split.
            result = extract_n_from_chunk_in_column(col_name)
    return result


def extract_Ns(columns):
    all_but_the_first = list(set(columns) -
                             POINTS_PERC_FEATURES_)
    # retrieve the different Ns.
    Ns = list(set(map(map_fn_N_only, all_but_the_first)))
    Ns = sorted(Ns)
    return Ns, all_but_the_first


def positive_filter_and_merge(df: pd.DataFrame, patterns: typing.Optional[typing.Iterable[str]] = None) -> pd.DataFrame:
    """
    Filter df on each pattern and merge the result.

    That is, for each pattern, we filter the input pd.DataFrame and merge it against the pd.DataFrame filtered
    using the previous pattern.
    :param df:
    :param patterns:
    :return:
    """
    if patterns is None:
        warnings.warn('positive_filter_and_merge got filter.')
        return df

    target = pd.DataFrame()
    for pattern in patterns:
        target = pd.merge(left=target, right=df.filter(like=pattern), left_index=True, right_index=True, how='outer')
    return target


class PositiveFilterAgg(enum.Enum):
    AND = 'AND'
    OR = 'OR'


def positive_filter_and_merge_2(df: pd.DataFrame, agg: PositiveFilterAgg, axis=int|str,
                                patterns: typing.Optional[typing.Iterable[str]] = None) -> pd.DataFrame:
    if patterns is None:
        warnings.warn('positive_filter_and_merge_2 got filter.')
        return df


    if agg == PositiveFilterAgg.AND:
        target = pd.DataFrame()
        for pattern in patterns:
            target = pd.merge(left=target, right=df.filter(like=pattern), left_index=True, right_index=True,
                              how='outer')
        result = target
    else:
        accu = [df.filter(like=pattern, axis=axis) for pattern in patterns]
        result = pd.concat(accu)
    return result

def extract_columns_by_metrics_and_n(columns: typing.Iterable[str], metrics: typing.Iterable[str],
                                     n: typing.Optional[int] = None, Ns: typing.Optional[typing.List[int]] = None
                                     ) -> typing.List[str]:
    """
    Extract the columns in `columns` matching the given metrics and N.

    :param Ns:
    :param columns:
    :param metrics:
    :param n:
    :return:
    """
    result = []
    for metric in metrics:
        for col in columns:

            # different match according to N.
            if n is not None:
                to_search = f'N{n}' if Ns is None else f'{n:0{get_N_digits(Ns)}d}'
                matched = f'{to_search}_AVG({metric})' in col
            else:
                matched = 'MONO_' in col and f'AVG({metric})' in col
            if matched:
                result.append(col)
    return result


def extract_pipelines(columns: typing.Iterable[str]):
    """
    Extract the names of the pipelines included in `columns`.
    :param columns:
    :return:
    """
    all_but_the_first = list(set(columns) -
                             POINTS_PERC_FEATURES_)
    pipelines = list(set(map(map_fn_pipeline_only, all_but_the_first)))
    pipelines = sorted(pipelines)
    return pipelines


def map_fn_pipeline_only(col_name: str) -> str:
    # the first part BEFORE n{int} is the one we are interested in.
    try:
        pipeline_name = col_name[:col_name.index(' N')]
    except ValueError:
        pipeline_name = col_name[:col_name.index('_AVG(')]
    return pipeline_name


class ComputedAnalysisType(enum.Enum):
    INTEGRAL = 'INTEGRAL'
    AVG ='AVG'

    def prefix(self) -> str:
        return self.value.lower()


class ModelType(enum.Enum):
    ORACLE = 'ORACLE'
    VANILLA = 'VANILLA'

    @staticmethod
    def from_name_in_file(val: str) -> "ModelType":
        if ModelType.ORACLE.value.lower() in val:
            return ModelType.ORACLE
        return ModelType.VANILLA


class DeltaType(enum.Enum):
    DELTA_REF ='DELTA_REF'
    DELTA_SELF ='DELTA_SELF'
    MODEL_QUALITY = 'MODEL_QUALITY'

    @staticmethod
    def from_name_in_file(val: str) -> "DeltaType":
        if DeltaType.DELTA_REF.to_name_in_file() in val:
            return DeltaType.DELTA_REF
        elif DeltaType.DELTA_SELF.to_name_in_file() in val:
            return DeltaType.DELTA_SELF
        else:
            return DeltaType.MODEL_QUALITY

    def to_name_in_file(self) -> str:
        if self == DeltaType.MODEL_QUALITY:
            return 'quality'
        else:
            return self.value.lower()


def is_in_patterns(single: str, patterns: typing.List[str]) -> bool:
    found = False
    for pattern in patterns:
        if pattern in single:
            found = True
            break
    return found


def compute_summary(df: pd.DataFrame,
                    col_x: typing.Optional[str] = None) -> typing.Dict[ComputedAnalysisType, pd.Series]:
    """
    Assumes that df contains *all* the columns to be used.
    :param df:
    :param col_x:
    :return:
    """
    col_x = col_x if col_x else const.KEY_PERC_DATA_POINTS

    results_integral = []
    cols_to_use = sorted(list(set(df.columns) - {const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES}))

    sub_df = df[cols_to_use]

    for col in sub_df.columns:
        x = df[col_x]
        y = sub_df[col]
        results_integral.append(integrate.trapezoid(x=x, y=y))

    return {
        ComputedAnalysisType.INTEGRAL: pd.Series(results_integral, index=cols_to_use),
        ComputedAnalysisType.AVG: sub_df.mean()
    }

STAT_SUMMARY_DF_FIRST_COL = 'Metric'

def merge_summary(summary: typing.Dict[ComputedAnalysisType, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame({k: summary[k] for k in summary.keys()})
    df.index.name = STAT_SUMMARY_DF_FIRST_COL
    return df


def read_stat_summary(path_or_df: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df

    df.set_index(STAT_SUMMARY_DF_FIRST_COL, inplace=True)
    # now we rename the two columns.
    new_cols = []
    for col in df.columns:
        for computation_type in ComputedAnalysisType:
            if computation_type.value == col:
                new_cols.append(computation_type)
                break
    # add the first column which is the index name
    # new_cols = [STAT_SUMMARY_DF_FIRST_COL] + new_cols
    df.columns = new_cols
    return df


def build_str_label_from_stats(column: str, df: pd.DataFrame) -> str:
    result = column
    for single_index in df.index:
        # print(f'{single_index}, {column}? {single_index == column}')
        if single_index == column:
            accu = []
            series = df.loc[single_index]
            for computation_type in ComputedAnalysisType:
                # :3 because we don't want it to be too long.
                accu.append(f'{computation_type.value[:3]}: {round(series[computation_type],3)}')
            result = f'{result}[{", ".join(accu)}]'
            break
    # print(f'Input: {column}, output: {result}')
    return result


def get_stat_summary_df_from_args(args) -> typing.Optional[pd.DataFrame]:
    stat_summary_df = None
    if args.stat_summary_file is not None:
        stat_summary_df = read_stat_summary(path_or_df=args.stat_summary_file)
    return stat_summary_df


def get_recursive_merge_kwargs(target: pd.DataFrame):
    if len(target) == 0:
        # if empty, we cannot merge on the percentage of poisoning.
        kwargs = {'left_index': True, 'right_index': True, 'how': 'outer'}
    else:
        kwargs = {'on': list(POINTS_PERC_FEATURES_)}
    return kwargs


@dataclasses.dataclass
class PipelineAgainst(mashumaro.DataClassDictMixin):
    col_name: str
    prefix_to_use_for_export: str


@dataclasses.dataclass
class PipelineNamesToMatch(mashumaro.DataClassDictMixin):
    # "our" model
    other: str
    # against some baselines
    against: typing.List[PipelineAgainst]
    plot_prefix: str


def read_patterns(patterns_file: typing.Optional[str] = None,
                  patterns: typing.Optional[typing.List[PipelineNamesToMatch]] = None) -> typing.List[PipelineNamesToMatch]:
    if patterns_file is None and patterns is None:
        raise ValueError('Provide either patterns_file or patterns')

    if patterns_file is not None:
        with open(patterns_file, 'r') as f:
            raw_patterns = f.read()
    else:
        raw_patterns = patterns

    # patterns = decoder.decode(raw_patterns)
    parsed_patterns = json.loads(raw_patterns)
    result = [PipelineNamesToMatch.from_dict(v) for v in parsed_patterns]
    # return patterns
    return result


def add_basic_info_to_df(sub_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    sub_df[const.KEY_PERC_DATA_POINTS] = full_df[const.KEY_PERC_DATA_POINTS]
    sub_df[const.KEY_PERC_FEATURES] = full_df[const.KEY_PERC_FEATURES]
    return sub_df


def index_renamer(col_name: str, mapper: typing.Dict[int, str]) -> str:
    """
    Rename columns containing 'N1' with 'N01' as an example.
    :param col_name:
    :param mapper:
    :return:
    """
    try:
        n_split_before = col_name.index(' N')
    except ValueError:
        # if not found, return the string as is.
        return col_name

    n_split_after = col_name.index('_AVG')
    n = int(col_name[n_split_before:n_split_after].replace('N', ''))
    return col_name.replace(f'N{n}', f'N{mapper[n]}')


def get_N_digits(Ns: typing.List[int]) -> int:
    return len(str(Ns[-1]))


def transpose_arbitrary_delta_ref(df: pd.DataFrame) -> pd.DataFrame:
    """
    the DataFrame *mus* be shaped as follows:
    # metric    value1  value2  ... value_n
    # where metric has values such as A_vs_B_AVG(Recall)
    :param df:
    :return:
    """
    # the first thing we do is sort the index, replacing N1 with, for instance, N01
    # so that the sort works properly.
    Ns, _ = extract_Ns(df.index.to_list())
    Ns = sorted(Ns)
    # build a mapping.
    formatted = {n: f'{n:0{get_N_digits(Ns)}d}' for n in Ns}
    df.rename(lambda ind: index_renamer(ind, mapper=formatted), axis='index', inplace=True)

    groups = []

    # split by the metric
    grouped = df.groupby(by=lambda idx: idx.split('_')[-1])
    for metric_name, group in grouped:
        # metric name is like 'AVG(Accuracy)' while group
        # is the corresponding pd.DataFrame
        # now, basically, we modify `group` by updating the value of
        # the index, e.g., from 'A_vs_B N21_AVG(Recall)` to
        #  'A_vs_B N21`.
        group.rename(lambda ind: ind[:ind.index('_AVG')], axis='index', inplace=True)
        # now, we rename also the metrics, e.g., the value of the integral.
        group.rename(lambda col: f'{metric_name}({col.prefix() if isinstance(col, ComputedAnalysisType) else col})',
                     axis='columns', inplace=True)
        groups.append(group)

    # and now we put all pd.DataFrame together, using a join.
    df = utils.merge_multiple(dfs=groups, mask=[(False, '') for _ in range(len(groups))], on=df.index.name
                                ).sort_index()

    def key_fn(cols):
        # the key_fn must be vectorized
        # the close-open bracket indicates the very last part of the column, i.e., (avg) or (integral)
        return [col_name[col_name.index(')(')+1:] for col_name in cols]

    # at the very end, we sort the columns to avoid having accuracy_avg, accuracy_integral, ...
    # and instead we want to have accuracy_integral, recall_integral, precision_integral,
    # accuracy_avg, ...
    df.sort_index(axis='columns', inplace=True, key=key_fn)
    return df


def get_ymin_and_ymax(df: pd.DataFrame):
    y_min_pre, y_max_pre = df.min().min(), df.max().max()
    # we use abs because if y_min_pre is negative, we will instead obtain an increment
    # which is not what we want, since we want to stay *below* the minimum.
    y_min = y_min_pre - (abs(y_min_pre) * 0.1)
    # not sure why I need to do it twice.
    y_max = y_max_pre + (abs(y_max_pre) * 0.1)
    return y_min, y_max


def filter_columns_and_metrics(df: pd.DataFrame,
                               columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                columns_patterns_to_include: typing.Optional[typing.List[str]] = None,
metrics: typing.Optional[typing.List[str]] = None,
                               keep_perc_data_points_and_features: bool = True
                               ) -> typing.List[str]:
    metrics = metrics or [const.METRIC_NAME_ACCURACY, const.METRIC_NAME_RECALL, const.METRIC_NAME_PRECISION]

    columns = utils.filter_columns(columns=df.columns, patterns_to_exclude=columns_patterns_to_exclude,
                                   patterns_to_include=columns_patterns_to_include)
    cols = [col for col in columns if is_in_patterns(col, metrics)]

    if keep_perc_data_points_and_features:
        cols = [const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES] + cols

    return cols


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')