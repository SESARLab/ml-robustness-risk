import os
import typing

import numpy as np
import pandas as pd

import const


def gen_datasets(initial: pd.DataFrame, n_data_points: typing.List[int], output_dir: str,
                 rng: typing.Optional[int] = None):
    # drop the column 'Poisoned'.
    initial.drop(columns=[const.COORD_POISONED], axis='columns', inplace=True)
    initial = initial.astype({const.COORD_LABEL: 'int8'})

    to_choose_from = np.arange(len(initial))

    max_digits = len(str(max(n_data_points)))

    for single_n_data_point in n_data_points:
        idx = np.random.default_rng(seed=rng).choice(to_choose_from, single_n_data_point)

        initial.iloc[idx].to_csv(os.path.join(output_dir, f'{single_n_data_point:0{max_digits}}.csv'), index=False)


def top_gen_datasets(args):
    gen_datasets(pd.read_csv(args.input_dataset), n_data_points=args.n_data_points,
                 output_dir=args.output_dir, rng=args.rng)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers()
    parser_gen_datasets = sub_parsers.add_parser('gen-datasets')
    parser_gen_datasets.add_argument('--input-dataset', type=str, required=True)
    parser_gen_datasets.add_argument('--output-dir', type=str, required=True)
    parser_gen_datasets.add_argument('--rng', type=int, default=None, required=False)
    parser_gen_datasets.add_argument('--n-data-points', type=int, required=True, nargs='+')
    parser_gen_datasets.set_defaults(func=top_gen_datasets)

    args_ = parser.parse_args()
    args_.func(args_)
