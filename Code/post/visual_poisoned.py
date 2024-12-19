import enum
import typing

import json5 as json

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import decomposition, manifold

import const
import utils


class Method(enum.Enum):
    TSNE = 'TSNE'
    PCA = 'PCA'

    def to_method(self):
        if self == Method.TSNE:
            return manifold.TSNE(n_components=2, init='pca')
        elif self == Method.PCA:
            return decomposition.PCA(n_components=2)


def embed_dataset(clean: pd.DataFrame, method: Method, poisoned: typing.Dict[str, typing.Dict[str, pd.DataFrame]],):

    # embed the clean dataset.
    clean = utils.df_negate_filter(clean, patterns_to_exclude=[const.COORD_LABEL, const.COORD_POISONED])
    # print(clean.head())

    X_embedded = method.to_method().fit_transform(clean.values)

    attacks_name = list(poisoned.keys())
    percentages_name = list(poisoned[attacks_name[0]].keys())

    # build the overall df
    df_overall = pd.DataFrame({'d1': X_embedded[:, 0], 'd2': X_embedded[:, 1]})
    for attack_name, poisoned_datasets in poisoned.items():
        df_poisoned = pd.DataFrame({f'{attack_name}_{poisoned_dataset_name}': poisoned_dataset[const.COORD_POISONED].astype(int)
                                        for poisoned_dataset_name, poisoned_dataset in poisoned_datasets.items()})
        df_overall = df_overall.join(df_poisoned)

    n_rows = len(percentages_name) + 1
    n_cols = len(attacks_name)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10 * n_cols, 10 * n_rows))

    for attack_name_idx, (attack_name, poisoned_datasets) in enumerate(poisoned.items()):
        # the first row contain clean data only.
        axs[0, attack_name_idx].scatter(df_overall['d1'], df_overall['d2'], color='green')
        axs[0, attack_name_idx].set_title(f'clean')

        # now, we print clean and poisoned data for each attack.
        # enumerate starts at 1 because the first row is already occupied
        for poisoned_dataset_idx, (poisoned_dataset_name, poisoned_dataset) in enumerate(poisoned_datasets.items(), 1):
            # and here we plot data.
            X_clean = df_overall[df_overall[f'{attack_name}_{poisoned_dataset_name}'] == 0]
            X_poisoned = df_overall[df_overall[f'{attack_name}_{poisoned_dataset_name}'] == 1]

            axs[poisoned_dataset_idx, attack_name_idx].scatter(X_clean['d1'], X_clean['d2'], color='green')
            axs[poisoned_dataset_idx, attack_name_idx].scatter(X_poisoned['d1'], X_poisoned['d2'], color='red')
            axs[poisoned_dataset_idx, attack_name_idx].set_title(f'{attack_name}_{poisoned_dataset_name}')
    # plt.show()
    return df_overall, fig


def read_config(config_file_path: str):
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    df_clean = pd.read_csv(config['clean'])
    df_poisoned = {attack_name: {poisoned_dataset_name: pd.read_csv(poisoned_dataset_path)
                                 for poisoned_dataset_name, poisoned_dataset_path in config['poisoned'][attack_name].items()}
                   for attack_name in config['poisoned']}

    # just some safety checks
    for attack_name, poisoned_datasets in df_poisoned.items():
        duplicates = utils.get_duplicates(list(poisoned_datasets.keys()))
        if len(duplicates) > 0:
            raise ValueError(f'There duplicated in {attack_name}: {duplicates}')
    return df_clean, df_poisoned


def top_embed(args):
    if args.output_file_csv is None and args.output_file_plot is None:
        raise ValueError('At least one between output_file_csv and output_file_plot must be specified')

    df_clean, df_poisoned = read_config(args.config_file)
    df, fig = embed_dataset(clean=df_clean, poisoned=df_poisoned, method=Method[args.method],)

    if args.output_file_plot is not None:
        fig.savefig(args.output_file_plot, dpi=200)
    if args.output_file_csv is not None:
        df.to_csv(args.output_file_csv, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    parser_embed = sub_parsers.add_parser('embed')
    parser_embed.add_argument('--config-file', required=True, type=str)
    parser_embed.add_argument('--method', required=True, choices=[how.value for how in Method],)
    parser_embed.add_argument('--output-file-csv', required=False, type=str)
    parser_embed.add_argument('--output-file-plot', required=False, type=str)
    parser_embed.set_defaults(func=top_embed)

    args_ = parser.parse_args()
    args_.func(args_)
