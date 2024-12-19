import json
import typing

import pandas as pd

SORT_KEY = ['N', 'group', 'step_type', 'routing', 'n_data_points']

def parse_mono(data: dict) -> pd.DataFrame:
    value_s = []
    benchmarks = data['benchmarks']
    for benchmark in benchmarks:
        # just grab the number of data points and the corresponding value
        value_s.append({
            'n_data_points': benchmark['extra_info']['n_data_points'],
            'Time(s)': benchmark['stats']['mean'],
        })

    return pd.DataFrame(value_s).sort_values(by=['n_data_points'])


def parse_ensemble(data: dict) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    value_s = []
    benchmarks = data['benchmarks']
    for benchmark in benchmarks:
        value_s.append({
            'n_data_points': benchmark['extra_info']['n_data_points'],
            'group': benchmark['extra_info']['pipeline_group'],
            'step_type': benchmark['extra_info']['pipeline_step_type'],
            'routing': benchmark['extra_info']['pipeline_routing'],
            'N': benchmark['extra_info']['pipeline_n'],
            'Time(s)': benchmark['stats']['mean'],
        })

    df_all = pd.DataFrame(value_s)

    # now basically we group by 'group', 'step_type', and 'n', to retrieve the worst
    # config for each setting.
    # this creates a "fictional" DataFrame where the index is the set of grouped-over columns,
    # and the columns are the *maximum for each column*, in each group.
    df_worst = df_all.groupby(['n_data_points', 'group', 'step_type', 'N']).max().reset_index()

    return df_all.sort_values(by=SORT_KEY), df_worst.sort_values(by=SORT_KEY)


def merge_mono_ensemble(mono: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    # a successful merge requires joint columns.
    mono['group'] = 'MONO'
    mono['step_type'] = 'MONO'
    mono['routing'] = 'MONO'
    mono['N'] = 1

    merged = pd.concat([mono, ensemble])

    return merged.sort_values(by=SORT_KEY)


# def prepare_for_plot(df: pd.DataFrame) -> pd.DataFrame:



def top_parse_all(args):
    with open(args.input_file_mono) as f:
        data = json.loads(f.read())
    result_mono = parse_mono(data=data)

    with open(args.input_file_ensemble) as f:
        data = json.loads(f.read())
    result_ensemble = parse_ensemble(data=data)

    merged = merge_mono_ensemble(mono=result_mono, ensemble=result_ensemble[0])

    for out_df, out_file_name_postfix in [
        (result_mono, 'mono.csv'),
        (result_ensemble[0], 'ensemble.csv'),
        (result_ensemble[1], 'ensemble_worst.csv'),
        (merged, 'all.csv'),
    ]:
        out_df.to_csv(f'{args.output_file_prefix}_{out_file_name_postfix}', index=False)


def top_parse_ensemble_or_risk(args):
    with open(args.input_file) as f:
        data = json.loads(f.read())
    result_ensemble = parse_ensemble(data=data)
    for out_df, out_file_name_postfix in [
        (result_ensemble[0], 'risk.csv'),
        (result_ensemble[1], 'risk_worst.csv'),
        ]:
        out_df.to_csv(f'{args.output_file_prefix}_{out_file_name_postfix}', index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers()

    parser_all = sub_parsers.add_parser('parse-all')
    parser_all.add_argument('--input-file-mono', type=str, required=True)
    parser_all.add_argument('--input-file-ensemble', type=str, required=True)
    parser_all.add_argument('--output-file-prefix', type=str, required=True)
    parser_all.set_defaults(func=top_parse_all)

    parser_risk = sub_parsers.add_parser('parse-risk')
    parser_risk.add_argument('--input-file', type=str, required=True)
    parser_risk.add_argument('--output-file-prefix', type=str, required=True)
    parser_risk.set_defaults(func=top_parse_ensemble_or_risk)

    args_ = parser.parse_args()
    args_.func(args_)