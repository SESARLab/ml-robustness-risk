import typing

import numpy as np
import pandas as pd
import xarray as xr

import const
from experiments.base import ExpInfo


def make_3d_array(avg_array: xr.DataArray, std_array: xr.DataArray, coord_perc_points: typing.Sequence[float],
                  coord_perc_features: typing.Sequence[float],
                  poisoned_idx: typing.Optional[typing.Union[typing.Sequence[int], np.ndarray]] = None) -> xr.DataArray:
    coords_to_assign = {
        const.COORD_PERC_POINTS: (['x'], coord_perc_points),
        const.COORD_PERC_FEATURES: (['x'], coord_perc_features),
        'z': (['z'], [const.KEY_COORD_Z_AVG, const.KEY_COORD_Z_STD])
    }
    if poisoned_idx is not None:
        coords_to_assign[const.COORD_POISONED] = (['x'], poisoned_idx)

    return xr.concat([avg_array, std_array], dim='z'
                     ).assign_coords(coords_to_assign)


def merge_3d_array(arr: xr.DataArray) -> xr.DataArray:
    avg = arr.sel(z=const.KEY_COORD_Z_AVG).assign_coords(
        {'y': [f'{const.KEY_COORD_Z_AVG}({coo})' for coo in arr.coords['y'].values]})
    std = arr.sel(z=const.KEY_COORD_Z_STD).assign_coords(
        {'y': [f'{const.KEY_COORD_Z_STD}({coo})' for coo in arr.coords['y'].values]})
    # z is no longer needed
    return xr.concat([avg, std], dim='y').drop_vars('z')


def add_perc_points_and_features(arr: xr.DataArray) -> xr.DataArray:
    perc_points = arr.coords['x'][const.COORD_PERC_POINTS].values
    perc_features = arr.coords['x'][const.COORD_PERC_FEATURES].values
    # usual reshape to match target shape and properly append a new column.
    return xr.concat([xr.DataArray(perc_points.reshape(-1, 1), dims=('x', 'y'), coords={'y': [const.COORD_PERC_POINTS]}),
                      xr.DataArray(perc_features.reshape(-1, 1), dims=('x', 'y'), coords={'y': [const.COORD_PERC_FEATURES]}),
                      arr
                      ], dim='y', combine_attrs='no_conflicts')


def merge_and_add_all(arr: xr.DataArray, name: str) -> xr.DataArray:
    arr = merge_3d_array(arr)
    arr = add_perc_points_and_features(arr)
    arr = add_pipeline_name(arr, name)
    return arr


def add_pipeline_name(arr: xr.DataArray, name: str) -> xr.DataArray:
    return xr.concat([xr.DataArray(np.repeat(name, len(arr)).reshape(-1, 1), dims=('x', 'y'),
                                   coords={'y': [const.KEY_COORD_PIPELINE_NAME]}),
                      arr], dim='y')


def data_array_to_df(arr: xr.DataArray) -> pd.DataFrame:
    return pd.DataFrame(arr.to_numpy(), columns=arr.coords['y'].values)


def merge_and_rename(df1: pd.DataFrame, df2: pd.DataFrame, func_df1: typing.Callable[[str], str],
                     func_df2: typing.Callable[[str], str]):
    df1 = df1.rename(func_df1, axis='columns')
    df2 = df2.rename(func_df2, axis='columns')
    return pd.merge(df1, df2, left_index=True, right_index=True, validate='one_to_one')


def merge_repeatedly(dfs: typing.Sequence[pd.DataFrame], pipeline_names: typing.Sequence[str]) -> pd.DataFrame:
    merged = dfs[0]
    merged = merged.rename(lambda col: f'{pipeline_names[0]}_{col}', axis='columns')

    for i in range(1, len(dfs)):
        merged = merge_and_rename(df1=merged, df2=dfs[i], func_df1=lambda col: col,
                                  func_df2=lambda col: f'{pipeline_names[i]}_{col}')
    return merged


def merge_repeatedly_and_drop_unnecessary_columns(dfs: typing.Sequence[pd.DataFrame],
                                                  pipeline_names: typing.Sequence[str],
                                                  drop_std: bool = False):
    merged = dfs[0]
    merged = merged.rename(lambda col: f'{pipeline_names[0]}_{col}'

    if col not in const.INFO_KEY_LIST else col,
                           axis='columns')
    # then we drop the column with the pipeline name. It is not necessary since
    # we are already renaming every column with the pipeline name.
    merged = merged.drop(columns=[const.KEY_PIPELINE_NAME])

    for i in range(1, len(dfs)):
        to_merge = dfs[i]
        # here we drop the information related to pipeline name, perc points, and so on,
        # because they are the same and already added in the first pd.DataFrame.
        to_merge = to_merge.drop(columns=[const.KEY_PIPELINE_NAME, const.KEY_PERC_DATA_POINTS, const.KEY_PERC_FEATURES])

        merged = merge_and_rename(df1=merged, df2=to_merge, func_df1=lambda col: col,
                                  func_df2=lambda col: f'{pipeline_names[i]}_{col}')

    if drop_std:
        # now we remove all the columns related to std if required.
        # to_drop = [col for col in merged.columns if PREFIX_STD in col]
        # merged = merged.drop(columns=to_drop)
        return drop_std_from_df(df=merged)
    return merged


def xr_to_df(arr: xr.DataArray) -> pd.DataFrame:
    if len(arr.dims) != 2:
        raise ValueError('This function works with 2d array only.')
    # vanilla conversion does not work, apparently.
    data = arr.to_numpy()
    cols = arr.coords['y'].values
    return pd.DataFrame(data=data, columns=cols)


def xr_to_df_with_poisoned_x(arr) -> pd.DataFrame:
    arr = arr.rename({'x': const.COORD_POISONED})
    got = xr_to_df(arr)
    got.index = arr.coords[const.COORD_POISONED].values
    return got


def drop_std_from_df(df: pd.DataFrame) -> pd.DataFrame:
    # now we remove all the columns related to std if required.
    to_drop = [col for col in df.columns if const.PREFIX_STD in col]
    df = df.drop(columns=to_drop)
    return df


def just_merge_repeatedly(dfs: typing.Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = merge_and_rename(df1=merged, df2=dfs[i], func_df1=lambda col: col, func_df2=lambda col: col)
    return merged


def df_mean_and_std(df: pd.DataFrame, numeric_only: bool = True):
    """

    :param df:
    :param numeric_only
    :return: pd.Series
    """

    avg = df.mean(axis='rows', numeric_only=numeric_only)
    std = df.std(axis='rows', numeric_only=numeric_only)

    return pd.concat([avg.rename(lambda col: f'{const.PREFIX_AVG}({col})'),
                      std.rename(lambda col: f'{const.PREFIX_STD}({col})')])


def df_mean_and_std_drop_and_add_info(df: pd.DataFrame, info: ExpInfo) -> pd.Series:
    df = df.drop(const.KEY_PIPELINE_NAME, axis='columns')
    df = df.drop(const.KEY_PERC_DATA_POINTS, axis='columns')
    df = df.drop(const.KEY_PERC_FEATURES, axis='columns')

    merged = df_mean_and_std(df)

    return info.prepend_to(merged)
