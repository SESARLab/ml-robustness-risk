import dataclasses
import os
import typing
import warnings

import joblib
import numpy as np
import xarray as xr
# import xarray_extras.csv
from sklearn import utils as sk_utils

import const
import utils
import utils_exp_post
from . import base
from poisoning import wrapper, generator


def xr_and_df_from_X_y(X, y, columns):
    dataset_xr = xr.DataArray(np.hstack([X, y.reshape(-1, 1)]),
                                         dims=('x', 'y'), coords={'y': columns + [const.COORD_LABEL]})
    dataset_df = utils_exp_post.xr_to_df(dataset_xr)
    return dataset_xr, dataset_df

@dataclasses.dataclass
class DatasetGenerator:
    # X: np.ndarray
    # y: np.ndarray
    X_train_clean: np.ndarray
    y_train_clean: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    _poisoning_input: generator.PoisoningGenerationInput

    columns: typing.Optional[typing.List[str]] = dataclasses.field(default=None)

    _poisoning_algos: typing.List[wrapper.Poisoning] = dataclasses.field(default_factory=list)
    _poisoning_points: typing.List[typing.Tuple[float, float]] = dataclasses.field(default_factory=list)
    _all_datasets: xr.Dataset = dataclasses.field(default_factory=xr.Dataset)

    def __len__(self):
        return len(self._all_datasets)

    @property
    def all_datasets(self) -> xr.Dataset:
        return self._all_datasets

    @property
    def poisoning_algos(self) -> typing.Sequence[wrapper.Poisoning]:
        return self._poisoning_algos

    def __post_init__(self):
        # sk_utils.check_X_y(self.X, self.y)
        sk_utils.check_X_y(self.X_train_clean, self.y_train_clean)
        sk_utils.check_X_y(self.X_test, self.y_test)

        # check the precision of the percentage of points/features to poison.
        # More than 3 decimal numbers will cause issues in export.
        for single_perc_point in self._poisoning_points:
            for single_perc in single_perc_point:
                s = len(str(single_perc).split('.')[1])
                if s > 3:
                    warnings.warn(f'The precision of poisoning point {single_perc_point} is > 3. '
                                  f'It will cause issues during export.')

        self.columns = utils.check_and_get_columns(expected=self.X_train_clean.shape[1], got=self.columns)

    @staticmethod
    def from_dataset_to_poison(X_train, y_train, X_test, y_test, poisoning_generation_input: generator.PoisoningGenerationInput):

        # X_train_clean, X_test, y_train_clean, y_test = model_selection.train_test_split(
        #     X, y, train_size=poisoning_generation_input.train_split,
        #     shuffle=poisoning_generation_input.shuffle)

        poisoning_points, wrappers = poisoning_generation_input.generate_from_sequence()

        # it's too early to create the multidimensional xr.DataArray.
        return DatasetGenerator(
            # X=X, y=y,
            X_train_clean=X_train, y_train_clean=y_train,
            X_test=X_test, y_test=y_test, _poisoning_points=poisoning_points, _poisoning_algos=wrappers,
            columns=poisoning_generation_input.columns, _poisoning_input=poisoning_generation_input
        )

    @staticmethod
    def from_dataset_already_poisoned_dataset(
            X_y_train_clean: xr.DataArray,
            X_y_test: xr.DataArray,
            # X_y_train_clean: xr.Dataset,
            # X_y_test: xr.Dataset,
                                              poisoned_datasets: xr.Dataset,
                                              poisoning_generation_input: generator.PoisoningGenerationInput
                                              ) -> "DatasetGenerator":
        # the column names are in y coords. We pick all but the one with value const.COORD_LABEL
        # that indicates the column with the label.
        columns = X_y_train_clean.coords['y'].values
        columns = columns[columns != const.COORD_LABEL]

        # X_train_clean: np.ndarray = X_y_train_clean.loc[:, columns].values
        # y_train_clean: np.ndarray = X_y_train_clean.loc[:, const.COORD_LABEL].values
        #
        # X_test: np.ndarray = X_y_test.loc[:, columns].values
        # y_test: np.ndarray = X_y_test.loc[:, const.COORD_LABEL].values
        X_train_clean = X_y_train_clean.loc[:, columns].to_numpy()
        y_train_clean = X_y_train_clean.loc[:, const.COORD_LABEL].to_numpy()

        X_test = X_y_test.loc[:, columns].to_numpy()
        y_test = X_y_test.loc[:, const.COORD_LABEL].to_numpy()

        # vstack when working with multidimensional arrays.
        # X = np.vstack([X_train_clean, X_test])
        # y = np.hstack([y_train_clean, y_test])

        poisoning_points, wrappers = poisoning_generation_input.generate_from_sequence()

        # check that all the percentages are found.
        for perc_point, perc_feature in poisoning_points:
            found = False
            for poisoned_dataset in poisoned_datasets.values():
                if poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS] == perc_point and \
                        poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES] == perc_feature:
                    found = True
            if not found:
                raise ValueError(f'Percentage of poisoning ({perc_point}, {perc_feature}) not found in poisoned '
                                 f'datasets. Got: '
                                 f'{[(poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS], poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS]) for poisoned_dataset in poisoned_datasets.values()]}')

        # load only those specified in the sequence.
        kept = []

        # now, we consider only the poisoning points that are specified in the input.
        for poisoned_dataset in poisoned_datasets.values():
            found = False
            for perc_point, perc_feature in poisoning_points:
                if poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS] == perc_point and \
                        poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES] == perc_feature:
                    found = True
            if found:
                kept.append(poisoned_dataset)

        poisoned_datasets_filtered = xr.Dataset(data_vars={
            (data_array.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS],
             data_array.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES]): data_array
            for data_array in kept})

        return DatasetGenerator(X_train_clean=X_train_clean, y_train_clean=y_train_clean,
                                X_test=X_test, y_test=y_test, _poisoning_points=poisoning_points,
                                _poisoning_algos=wrappers, # _all_datasets=poisoned_datasets,
                                _all_datasets=poisoned_datasets_filtered,
                                _poisoning_input=poisoning_generation_input)

    def clean_dataset_attrs(self) -> typing.Dict[str, typing.Dict[str, float]]:
        return {const.KEY_ATTR_POISONED: self._poisoning_algos[0].perform_info.get_info_clean_as_dict()}

    def generate(self) -> "DatasetGenerator":
        """
        Idempotent. If override is specified, a new instance is returned leaving the current one,
        if existing, untouched.

        :return:
        """

        def inner_loop(poisoning_algo_: wrapper.Poisoning) -> xr.DataArray:
            X_poisoned, y_poisoned = poisoning_algo_.fit(X=self.X_train_clean, y=self.y_train_clean
                                                         ).transform(X=self.X_train_clean, y=self.y_train_clean)

            # convert the index of poisoned data points [15, 30, ..., ] to a boolean index
            # that will be hstack-ed.
            poisoning_idx = np.zeros(len(X_poisoned))
            poisoning_idx[poisoning_algo_.performer.idx_of_poisoned_data_points] = True

            arr = xr.DataArray(np.hstack([X_poisoned, y_poisoned.reshape(-1, 1), poisoning_idx.reshape(-1, 1)]),
                               dims=('x', 'y'), coords={'y': self.columns + [const.COORD_LABEL, const.COORD_POISONED]},
                               attrs={const.KEY_ATTR_POISONED: poisoning_algo_.perform_info.get_info_as_dict()})
            return arr

        # with joblib.parallel_backend(backend='dask'):
        with joblib.Parallel(n_jobs=len(self._poisoning_algos)) as parallel:
            all_datasets: typing.List[xr.DataArray] = parallel(
                joblib.delayed(inner_loop)(poisoning_algo) for poisoning_algo in self._poisoning_algos)
        # all_datasets = [inner_loop(poisoning_algo) for poisoning_algo in self._poisoning_algos]

        poisoned_datasets = xr.Dataset(data_vars={
            (data_array.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS],
             data_array.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES]): data_array
                                                   for data_array in all_datasets})

        # if override and len(self._all_datasets) > 0:
        #     return DatasetGenerator(
        #         X=self.X, y=self.y, X_train_clean=self.X_train_clean, y_train_clean=self.y_train_clean,
        #         X_test=self.X_test, y_test=self.y_test, columns=self.columns,
        #         _poisoning_algos=self._poisoning_algos, _poisoning_points=self._poisoning_points,
        #         _all_datasets=poisoned_datasets, _poisoning_input=self._poisoning_input)
        # else:
        #     self._all_datasets = xr.Dataset(data_vars={data_array.attrs.values(): data_array
        #                                            for data_array in all_datasets})
        #     return self
        self._all_datasets = poisoned_datasets
        return self

    def export(self, base_directory: typing.Optional[str] = None, exists_ok: bool = False):
        if base_directory is not None:
            # create directory.
            os.makedirs(base_directory, exist_ok=exists_ok)

            base_directory_csv = os.path.join(base_directory, base.DIR_DATASET_NAME_EXPORT_CSV)
            os.makedirs(base_directory_csv, exist_ok=exists_ok)

            # clean_dataset = xr.DataArray(np.hstack([self.X_train_clean, self.y_train_clean.reshape(-1, 1)]),
            #                              dims=('x', 'y'), coords={'y': self.columns + [const.COORD_LABEL]})
            #
            # utils_exp_post.xr_to_df(clean_dataset).to_csv(
            #     os.path.join(base_directory_csv, f'{base.FILE_NAME_DATASET_PREFIX_CLEAN}.csv'), index=False)
            clean_dataset_xr, clean_dataset_df = xr_and_df_from_X_y(X=self.X_train_clean, y=self.y_train_clean,
                                                                    columns=self.columns)
            clean_dataset_df.to_csv(os.path.join(base_directory_csv, f'{base.FILE_NAME_DATASET_PREFIX_CLEAN}.csv'),
                                    index=False)

            for i, dataset_name in enumerate(self._all_datasets):
                utils_exp_post.xr_to_df(self._all_datasets[dataset_name]).to_csv(
                    os.path.join(base_directory_csv,
                                 f'{self._poisoning_points[i][0]}_{self._poisoning_points[i][1]}.csv'), index=False)

            # test_dataset = xr.DataArray(np.hstack([self.X_test, self.y_test.reshape(-1, 1)]),
            #                             dims=('x', 'y'), coords={'y': self.columns + [const.COORD_LABEL]})
            # # xarray_extras.csv.to_csv(test_dataset, os.path.join(base_directory_csv,
            # #                                                     f'{base.FILE_NAME_DATASET_PREFIX_TEST}.csv'), index=False)
            #
            # utils_exp_post.xr_to_df(test_dataset).to_csv(
            #     os.path.join(base_directory_csv, f'{base.FILE_NAME_DATASET_PREFIX_TEST}.csv'), index=False)

            test_dataset_xr, test_dataset_df = xr_and_df_from_X_y(X=self.X_test, y=self.y_test,
                                                                    columns=self.columns)
            test_dataset_df.to_csv(os.path.join(base_directory_csv, f'{base.FILE_NAME_DATASET_PREFIX_TEST}.csv'),
                                    index=False)

            base_directory_binary = os.path.join(base_directory, base.DIR_DATASET_NAME_EXPORT_BINARY)
            os.makedirs(base_directory_binary, exist_ok=exists_ok)
            # we finally export using xarray native format.
            # We export three datasets: poisoned, clean, and test.
            # Test has a different shape than poisoned and clean (shorter)
            # and cannot therefore be added to the others.
            # Clean has a different shape than poisoned (fewer columns)
            # and cannot therefore be added to the others.

            # Unfortunately, we need to do a bit of renaming because dictionary as key of the
            # dataset is not accepted.

            # we also need to "flatten" the attributes of each individual DataArray, we can't have a dict of dict.
            new_data_arr = {}
            for k in self._all_datasets.keys():
                new_k = from_poisoned_tuple_to_str(k)# from_poisoned_dict_to_str(k)
                # need to copy: otherwise we are changing the attributes of something that
                # should be not be changed since new_data_arr is only temporary.
                new_data_arr[new_k] = self._all_datasets[k].copy()
                new_data_arr[new_k].attrs = new_data_arr[new_k].attrs[const.KEY_ATTR_POISONED]
                # print(new_data_arr[new_k].attrs)

            xr.Dataset(new_data_arr).to_netcdf(os.path.join(base_directory_binary,
                                                      f'{base.FILE_NAME_DATASET_PREFIX_POISONED}.h5netcdf'),
                                         engine='h5netcdf')
            clean_dataset_xr.to_netcdf(os.path.join(base_directory_binary,
                                                 f'{base.FILE_NAME_DATASET_PREFIX_CLEAN}.h5netcdf'), engine='h5netcdf')
            test_dataset_xr.to_netcdf(os.path.join(base_directory_binary,
                                                f'{base.FILE_NAME_DATASET_PREFIX_TEST}.h5netcdf'), engine='h5netcdf')
            # we also export the configuration.

    @staticmethod
    def import_from_directory(base_directory: str,
                              poisoning_generation_input: generator.PoisoningGenerationInput) -> "DatasetGenerator":
        """

        Parameters
        ------
        base_directory: str
            It can be the directory containing both binary and non-binary datasets, or the subdirectory
            containing binary files only.
            To this aim, it looks whether binary files are there according to the expected names:
            -
            -
            -
            If not, it tries to go into a subdirectory named `base.DIR_DATASET_NAME_EXPORT_BINARY`.
            If not found, it raises a `ValueError`.

        :return:
        """
        files = os.listdir(base_directory)
        if base.DIR_DATASET_NAME_EXPORT_BINARY in files and \
                os.path.isdir(os.path.join(base_directory, base.DIR_DATASET_NAME_EXPORT_BINARY)):
            # ok, let's go to the binary directory
            base_directory = os.path.join(base_directory, base.DIR_DATASET_NAME_EXPORT_BINARY)

        # look for binary files.
        file_dataset_poisoned = os.path.join(base_directory, f'{base.FILE_NAME_DATASET_PREFIX_POISONED}.h5netcdf')
        file_dataset_test = os.path.join(base_directory, f'{base.FILE_NAME_DATASET_PREFIX_TEST}.h5netcdf')
        file_dataset_clean = os.path.join(base_directory, f'{base.FILE_NAME_DATASET_PREFIX_CLEAN}.h5netcdf')
        # files = list(map(lambda f: os.path.join(base_directory, f), files))
        # if file_dataset_poisoned not in files:
        #     raise ValueError(f'{base.DIR_DATASET_NAME_EXPORT_BINARY} missing ')

        X_y_train_clean = xr.open_dataarray(file_dataset_clean)
        X_y_test = xr.open_dataarray(file_dataset_test)
        # here, we need to rename the variable names back to tuple
        poisoned_datasets = xr.open_dataset(file_dataset_poisoned)
        poisoned_datasets = poisoned_datasets.rename({k: from_poisoned_str_to_tuple(k) for k in poisoned_datasets.keys()})
        # we also need to "de-flatten" the individual attributes as well, that have been flattened back for export.
        for k in poisoned_datasets.keys():
            poisoned_datasets[k].attrs = {const.KEY_ATTR_POISONED: poisoned_datasets[k].attrs}
        # need to import config. Maybe it works with dacite.
        # in reality the problem is significantly more complex: we need to export learnt model.
        # A thesis is necessary.

        # TODO clean and test.csv do not have the correct headers set

        return DatasetGenerator.from_dataset_already_poisoned_dataset(
            X_y_test=X_y_test, X_y_train_clean=X_y_train_clean, poisoned_datasets=poisoned_datasets,
            poisoning_generation_input=poisoning_generation_input)



# def from_poisoned_dict_to_str(val) -> str:
#     # what we receive is an object of instance dict_values (what dict.values() returns) that we must
#     # convert to a dict since we expect it to be a dict.
#     val = dict(list(val)[0])
#     return f'{val[const.COORD_PERC_POINTS]}_{val[const.COORD_PERC_FEATURES]}'

def from_poisoned_tuple_to_str(val: typing.Tuple[float, float]) -> str:
    return f'{val[0]}_{val[1]}'


# def from_poisoned_str_to_dict(val: str) -> typing.Dict[str, float]:
# #     splits = val.split('_')
# #
# #     return {
# #         const.COORD_PERC_POINTS: float(splits[0]),
# #         const.COORD_PERC_FEATURES: float(splits[1])
# #     }

def from_poisoned_str_to_tuple(val: str) -> typing.Tuple[float, float]:
    splits = val.split('_')
    return float(splits[0]), float(splits[1])


# IRRELEVANT_COLUMNS = {const.COORD_LABEL, base.COORD_POISONED}