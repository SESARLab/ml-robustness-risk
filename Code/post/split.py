import json
import typing

import numpy as np
import pandas as pd
from sklearn import model_selection

import const
import utils


def almost_top_split(path: str, train_test_split: float = .75) -> typing.Tuple[
    pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if const.COORD_LABEL not in df.columns:
        raise ValueError(f'Cannot find column \'{const.COORD_LABEL}\', got: {list(df.columns)}')

    all_but_label = [col for col in df.columns if col != const.COORD_LABEL]
    X = df[all_but_label].values
    y = df[[const.COORD_LABEL]].values.flatten()

    X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, train_size=train_test_split, shuffle=True)

    train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))),
                         columns=all_but_label + [const.COORD_LABEL])
    test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))),
                         columns=all_but_label + [const.COORD_LABEL])

    return train, test


def almost_top_describe(train_path: str, test_path: str):

    X_train, y_train = utils.load_dataset_from_csv(train_path)
    X_test, y_test = utils.load_dataset_from_csv(test_path)

    return {
        'TRAIN': {
            'LEN': len(X_train),
            'N_0': len(y_train[y_train == 0]),
            'N_1': len(y_train[y_train == 1]),
        },
        'TEST': {
            'LEN': len(X_test),
            'N_0': len(y_test[y_test == 0]),
            'N_1': len(y_test[y_test == 1]),
        }
    }


def top_split(args):
    train, test = almost_top_split(path=args.input_file, train_test_split=args.train_test_split)
    train.to_csv(path_or_buf=args.output_file_train, index=False)
    test.to_csv(path_or_buf=args.output_file_test, index=False)


def top_describe(args):
    desc = almost_top_describe(train_path=args.input_file_train, test_path=args.input_file_test)
    print(json.dumps(desc, indent=2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    parser_split = sub_parsers.add_parser('split')
    parser_split.add_argument('--input-file', type=str, required=True)
    parser_split.add_argument('--output-file-train', type=str, required=True)
    parser_split.add_argument('--output-file-test', type=str, required=True)
    parser_split.add_argument('--train-test-split', type=float, default=0.75)
    parser_split.set_defaults(func=top_split)

    parser_desc = sub_parsers.add_parser('describe')
    parser_desc.add_argument('--input-file-train', type=str, required=True)
    parser_desc.add_argument('--input-file-test', type=str, required=True)
    parser_desc.set_defaults(func=top_describe)

    args_ = parser.parse_args()
    args_.func(args_)
