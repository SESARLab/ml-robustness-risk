import dataclasses
import json5 as json
import os
import shutil
import typing

import joblib

from . import top


KEY_TARGET_DATASET = 'DATASET'
KEY_TARGET_EXP_ENSEMBLE_RISK = 'EXP_ENSEMBLE_RISK'
KEY_TARGET_EXP_ENSEMBLE_PLAIN_ADVANCED = 'EXP_ENSEMBLE_PLAIN_ADVANCED'
KEY_TARGET_EXP_MONOLITHIC_MODELS = 'EXP_MONOLITHIC_MODELS'
KEY_TARGET_EXP_IOP = 'EXP_IOP'
KEY_TARGET_LIST = [KEY_TARGET_DATASET,
                   KEY_TARGET_EXP_ENSEMBLE_RISK, KEY_TARGET_EXP_IOP,
                   KEY_TARGET_EXP_ENSEMBLE_PLAIN_ADVANCED, KEY_TARGET_EXP_MONOLITHIC_MODELS]
# the name of the key that must be in data and take one of the above values
KEY_TARGET_KEY = 'TARGET'


def from_conf(data: typing.Dict[str, typing.Any]
              ) -> typing.Union[top.TopLevelDataset, top.TopLevelExpEnsembleRisk, top.TopLevelExpIop]:
    if KEY_TARGET_KEY not in data:
        raise ValueError(f'Missing key \'{KEY_TARGET_KEY}\'')
    target = data[KEY_TARGET_KEY]
    if target not in KEY_TARGET_LIST:
        raise ValueError(f'Unknown value for \'{KEY_TARGET_KEY}\': {target}. Possible values: {KEY_TARGET_LIST}')
    if target == KEY_TARGET_DATASET:
        target_clazz = top.TopLevelDataset
    elif target == KEY_TARGET_EXP_IOP:
        target_clazz = top.TopLevelExpIop
    elif target == KEY_TARGET_EXP_ENSEMBLE_PLAIN_ADVANCED:
        target_clazz = top.TopLevelExpEnsemblePlainAdvanced
    elif target == KEY_TARGET_EXP_MONOLITHIC_MODELS:
        target_clazz = top.TopLevelExpMonolithicModels
    else:
        target_clazz = top.TopLevelExpEnsembleRisk
    data.pop(KEY_TARGET_KEY)
    return target_clazz.from_dict(data)


@dataclasses.dataclass
class ActionWrapper:
    target: typing.Union[top.TopLevelDataset, top.TopLevelExpEnsembleRisk, top.TopLevelExpIop]
    in_file: typing.Optional[str] = dataclasses.field(default=None)

    @staticmethod
    def sanity_checks(raw_content):
        # the goal here is to check if there are two duplicated keys one
        # within the other one.
        pass

    @staticmethod
    def from_file(file_path: str) -> "ActionWrapper":
        with open(file_path) as input_file:
            raw_content = input_file.read()
            input_file_parsed = json.loads(raw_content)
            parsed = from_conf(data=input_file_parsed)
            return ActionWrapper(target=parsed, in_file=file_path)

    def run_from_dask(self, max_worker: int, threads_per_worker: int = 1):
        import numpy as np
        np.seterr(invalid='raise')

        from joblib.externals.loky import set_loky_pickler

        set_loky_pickler('dill')

        with joblib.parallel_config(prefer='processes', n_jobs=1):
            self.target.do()

        if self.in_file is not None:
            shutil.copy(self.in_file, os.path.join(self.target.base_output_directory, 'input.jsonc'))
