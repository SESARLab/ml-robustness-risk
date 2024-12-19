import os
import typing

import numpy as np
import pandas as pd

import const
KEY_GROUP = 'GROUP'
KEY_PIPELINE = 'PIPELINE'
KEY_STEP_TYPE = 'STEP_TYPE'
KEY_ROUTING = 'ROUTING'
KEY_N = 'N'

KEY_GROUP_TSUSC = 'TSUSC'

ENV_INPUT_DATASET_DIR = 'INPUT_DATASET_DIR'

def map_fn(df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
    all_but_the_last = [col for col in df.columns if col != const.COORD_LABEL]
    return df[all_but_the_last].values, df[const.COORD_LABEL].values


def read_datasets(input_dir: str) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
    files = os.listdir(input_dir)
    files = map(lambda x: os.path.join(input_dir, x), files)
    return list(map(lambda x: map_fn(pd.read_csv(x)), files))
