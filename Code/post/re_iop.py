import json
import re

import pandas as pd
from matplotlib import pyplot as plt


def read_from(input_file: str, index: int):
    # https://stackoverflow.com/questions/67380802/after-saving-a-plotly-figure-to-html-file-can-you-re-read-it-later-as-a-figure
    with open(input_file) as f:
        html = f.read()
    # call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2 ** 16:])[0]
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_dict = {'data': call_args[1], 'layout': call_args[2]}

    # we need to locate the data we are interested in.
    # We are interested in two objects, the first is the green one, the second is the red one.
    # They are the two series we plot on the same graph
    # These two objects are "flat" in the array of data, first the green, then the red.
    # So we multiply by 2.
    indexes = [index*2, (index*2)+1]

    data_non_poisoned = plotly_dict['data'][indexes[0]]
    data_poisoned = plotly_dict['data'][indexes[1]]

    # now, we have to find the legend. So we look at the layout element
    legend_x = plotly_dict['layout']['xaxis']['title']['text']
    legend_y = plotly_dict['layout']['yaxis']['title']['text']

    # load data in a pd.DataFrame.
    # We need two pd.DataFrames because the length is not the same.
    df_clean = pd.DataFrame({legend_x: data_non_poisoned['x'], legend_y: data_non_poisoned['y']})
    df_poisoned = pd.DataFrame({legend_x: data_poisoned['x'], legend_y: data_poisoned['y']})
    # let's plot it, so that we can double-check
    fig, ax = plt.subplots()
    df_clean.plot.scatter(x=legend_x, y=legend_y, ax=ax, color='green')
    df_poisoned.plot.scatter(x=legend_x, y=legend_y, ax=ax, color='red')
    return df_clean, df_poisoned, fig



def top_load_and_export(args):
    read_from(args.input_file, index=args.index)
    df_clean, df_poisoned, fig = read_from(args.input_file, index=args.index)
    df_clean.to_csv(f'{args.output_file_prefix}_DATA_CLEAN.csv', index=False)
    df_poisoned.to_csv(f'{args.output_file_prefix}_DATA_POISONED.csv', index=False)
    fig.savefig(f'{args.output_file_prefix}_PIC.png', dpi=300)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers()

    parser_load = sub_parsers.add_parser('load')
    parser_load.add_argument('--input-file', type=str, required=True)
    parser_load.add_argument('--index', type=int, required=True)
    parser_load.add_argument('--output-file-prefix', type=str, required=True)
    parser_load.set_defaults(func=top_load_and_export)

    args_ = parser.parse_args()
    args_.func(args_)
