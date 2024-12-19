import typing

import pandas as pd

import const
from . import base
import utils


def almost_top_merge_files_standard(dfs: typing.List[pd.DataFrame],
                                    prefix_map: typing.List[str],
                                    columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                    columns_patterns_to_include: typing.Optional[typing.List[str]] = None,
                                    filter_as_union: bool = False) -> pd.DataFrame:
    dfs = list(map(lambda df: df[utils.filter_columns(columns=df.columns,
                                                      patterns_to_exclude=columns_patterns_to_exclude,
                                                      patterns_to_include=columns_patterns_to_include,
                                                      filter_as_union=filter_as_union)],
                   dfs))

    result: pd.DataFrame = utils.merge_multiple(dfs=dfs, on=list(base.POINTS_PERC_FEATURES_),
                                                mask=[(True, prefix) for prefix in prefix_map], )

    cols = result.columns.to_list()
    cols.remove(const.KEY_PERC_DATA_POINTS)
    cols.remove(const.KEY_PERC_FEATURES)
    cols = [const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES] + cols
    return result[cols]


def almost_top_merge_files_transposed(dfs: typing.List[pd.DataFrame],
                                      prefix_map: typing.List[str],
                                      column_as_index: str,
                                      columns_patterns_to_exclude: typing.Optional[typing.List[str]] = None,
                                      columns_patterns_to_include: typing.Optional[typing.List[str]] = None,
                                      rows_patterns_to_include: typing.Optional[typing.List[str]] = None,
                                      filter_as_union: bool = False,
                                      ) -> pd.DataFrame:
    # for this to work we need to set one column as index
    for i in range(len(dfs)):
        dfs[i] = dfs[i].set_index(column_as_index)

    dfs = list(map(lambda df: base.positive_filter_and_merge_2(
        df[utils.filter_columns(columns=df.columns,
                                patterns_to_exclude=columns_patterns_to_exclude,
                                patterns_to_include=columns_patterns_to_include,
                                filter_as_union=filter_as_union)],
        patterns=rows_patterns_to_include, agg=base.PositiveFilterAgg.OR, axis='rows'), dfs))

    result: pd.DataFrame = utils.merge_multiple(dfs=dfs, on=[column_as_index],
                                                mask=[(True, prefix) for prefix in prefix_map], )

    # we add a fake column we use for sorting.
    # we are just extracting the value of the index (index is a list, so we take the first item,
    # which is the string value.
    result['FAKE'] = result.apply(lambda row: row.index[0][row.index[0].index('_vs_'):], axis='rows')
    result.sort_values(by='FAKE', inplace=True)
    # now drop the column
    return result.drop(columns=['FAKE'])


def top_merge_files(args):
    if len(args.prefix) != len(args.input_files):
        raise ValueError('Prefix and input files must have the same length')

    dfs = [pd.read_csv(f) for f in args.input_files]
    result = almost_top_merge_files_standard(dfs=dfs, prefix_map=args.prefix,
                                             columns_patterns_to_exclude=args.columns_patterns_to_exclude,
                                             columns_patterns_to_include=args.columns_patterns_to_include,
                                             filter_as_union=args.patterns_combine_with_union_or_intersection)
    result.to_csv(args.output_file, index=False)


def to_merge_files_transposed(args):
    if len(args.prefix) != len(args.input_files):
        raise ValueError('Prefix and input files must have the same length')
    dfs = [pd.read_csv(f) for f in args.input_files]
    result = almost_top_merge_files_transposed(dfs=dfs, prefix_map=args.prefix,
                                               columns_patterns_to_exclude=args.columns_patterns_to_exclude,
                                               columns_patterns_to_include=args.columns_patterns_to_include,
                                               rows_patterns_to_include=args.rows_patterns_to_include,
                                               filter_as_union=args.patterns_combine_with_union_or_intersection,
                                               column_as_index=args.column_as_index
                                               )
    result.to_csv(args.output_file, index=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    parser.add_argument('--input-files', type=str, nargs='+', required=True, )
    parser.add_argument('--columns-patterns-to-exclude', type=str, nargs='+', required=False)
    parser.add_argument('--columns-patterns-to-include', type=str, nargs='+', required=False)
    parser.add_argument('--patterns-combine-with-union-or-intersection',
                                    type=base.str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--prefix', type=str, nargs='+', required=True)
    parser.add_argument('--output-file', type=str, required=True)

    parser_merge_files_standard = sub_parsers.add_parser('merge-files-standard')
    parser_merge_files_standard.set_defaults(func=top_merge_files)

    parser_merge_files_transposed = sub_parsers.add_parser('merge-files-transposed')
    parser_merge_files_transposed.add_argument('--rows-patterns-to-include', type=str, nargs='+',
                                               required=False)
    parser_merge_files_transposed.add_argument('--column-as-index', type=str, required=True)
    parser_merge_files_transposed.set_defaults(func=to_merge_files_transposed)

    args__ = parser.parse_args()
    args__.func(args__)
