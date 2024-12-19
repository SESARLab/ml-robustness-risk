import abc
import collections
import inspect
import itertools
import multiprocessing
import sys
import typing

import joblib
import numpy as np
import pandas as pd
from sklearn import base, metrics, multiclass, neighbors, pipeline, svm

import const

T = typing.TypeVar('T')
V = typing.TypeVar('V')


class ContainerRngAndJobsMixin:

    def __init__(self, rng: typing.Optional[int] = None,
                 n_jobs: typing.Optional[int] = None, *args, **kwargs):
        # first, forward arguments we don't need.
        super().__init__(*args, **kwargs)
        self.n_jobs = n_jobs
        if 'rng' in kwargs:
            kwargs.pop('rng')
        if 'n_jobs' in kwargs:
            kwargs.pop('n_jobs')
        if rng is not None:
            self.rs = np.random.RandomState(rng)
            self.rng = np.random.default_rng(rng)
        else:
            self.rs = None
            self.rng = np.random.default_rng()
        # self.rng = np.random.default_rng()
        # self.rs = np.random.random.__self__
        self.seed = rng


class EstimatorWrapperMixin(ContainerRngAndJobsMixin, typing.Generic[T]):

    def __init__(self, inner: typing.Type[T],
                 inner_kwargs, rng: typing.Optional[int] = None,
                 n_jobs: typing.Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if 'inner_kwargs' in kwargs:
        #     kwargs.pop('inner_kwargs')
        # if 'inner' in kwargs:
        #     kwargs.pop('inner')
        # super().__init__(*args, **kwargs)
        self.inner: T = inner(**inner_kwargs)

    def fit(self, X, y=None, fit_kwargs=None, **kwargs):
        if fit_kwargs is None:
            fit_kwargs = {}
        # find out if we need to include also y.
        signature = inspect.signature(self.inner.fit)
        if 'y' in signature.parameters.keys():
            fit_kwargs['y'] = y
        self.inner.fit(X, **fit_kwargs)
        # if hasattr(super, 'fit'):
        #     return super().fit(X, y, **kwargs)
        return self

    def transform(self, X, y=None, transform_kwargs=None, **kwargs):
        if transform_kwargs is None:
            transform_kwargs = {}
        return self.inner.predict(X, **transform_kwargs)


class EstimatorModelWrapper(EstimatorWrapperMixin):

    def predict_proba(self, X, **kwargs):
        return self.inner.predict_proba(X)

    def predict(self, X, **kwargs):
        return self.inner.predict(X)


class LabelMapper(EstimatorWrapperMixin):

    def __init__(self, cluster_centers: np.ndarray):
        super().__init__(inner=neighbors.KNeighborsTransformer, inner_kwargs={'n_neighbors': 10})
        self.cluster_centers = cluster_centers
        self.n_classes = len(cluster_centers)

    def fit(self, X, y=None, fit_kwargs=None, **kwargs):
        if self.n_classes != 2:
            raise ValueError('LabelMapper works with binary classification only')

        # y is the real one.
        super().fit(X=X, y=y)
        return self

    def transform(self, X, y):
        # if y.shape[1] != 2:
        #      raise ValueError(f'y.shape[1] != 2: got {y.shape}')

        y_real = y
        # y_cluster = y[:, 1]

        cluster_to_real = {}
        real_to_cluster = {}

        inner: neighbors.KNeighborsTransformer = self.inner
        neigh = inner.kneighbors([self.cluster_centers[0]], return_distance=False)

        # let's see its label
        neigh_real_labels = y_real[neigh]

        count_0 = np.count_nonzero(neigh_real_labels == 0)
        count_1 = np.count_nonzero(neigh_real_labels == 1)

        if count_0 >= count_1:
            real_to_cluster[0] = 0
            real_to_cluster[1] = 1
            cluster_to_real[0] = 0
            cluster_to_real[1] = 1
        else:
            real_to_cluster[1] = 0
            real_to_cluster[0] = 1
            cluster_to_real[0] = 1
            cluster_to_real[1] = 0

        assert len(cluster_to_real) == self.n_classes
        assert len(real_to_cluster) == self.n_classes

        # TODO this may need more tests but so far so good.

        return cluster_to_real, real_to_cluster


def get_n_jobs(n_jobs: typing.Optional[typing.Union[int, str]] = None):
    return -1 if n_jobs is not None and n_jobs == 'auto' else n_jobs


def min_core_count(seq: typing.Sequence):
    return min(len(seq), multiprocessing.cpu_count())


def merge_multiple(dfs: typing.Sequence[pd.DataFrame],
                   mask: typing.Sequence[typing.Tuple[bool, str]],
                   on: typing.Optional[typing.Union[str, list]] = None
                   ):
    """
    Utility function to merge multiple pd.DataFrame at once

    Parameters
    ---------
    dfs : typing.Sequence[pd.DataFrame] sequence of `DataFrame` to merge

    mask : typing.Sequence[typing.Tuple[bool, str]] sequence of the prefix/suffix to apply to
    the DataFrame (i.e., how to modify the columns for merging).
    Each element (tuple) of `mask` corresponds to a `DataFrame` in `dfs` according to its index.
    The first element indicates whether the second element must be applied as a prefix (`True`)
    or suffix (`False`).

    on : typing.Union[str, list] columns on which merge should be performed.

    Returns
    --------
    pd.DataFrame result of the merge


    Example
    >>>> import pandas as pd
    >>>>
    >>>> df1 = pd.DataFrame(data=[[11, 1, 2, 3], [12, 4, 5, 6]], columns=['I', 'A', 'B', 'C'])
    >>>> df2 = pd.DataFrame(data=[[11, 10, 20, 30], [12, 40, 50, 60]], columns=['I', 'A', 'B', 'C'])
    >>>> df3 = pd.DataFrame(data=[[11, 100, 200, 300], [12, 400, 500, 600]], columns=['I', 'A', 'B', 'C'])
    >>>>
    >>>> merged_ = merge_multiple([df1, df2, df3], mask=[(False, '_1'), (False, '_2'), (False, '_3')], on='I')
    >>>> merged_
        I  A_1  B_1  C_1  A_2  B_2  C_2  A_3  B_3  C_3
    0   11    1    2    3   10   20   30  100  200  300
    1   12    4    5    6   40   50   60  400  500  600
    """
    if len(dfs) != len(mask):
        raise ValueError(f'results ({len(dfs)}) and mask ({len(mask)}) must have the same length')

    def change_single(current_value: str, current_mask: str, before_or_after: bool):
        if before_or_after:
            return f'{current_mask}{current_value}'
        return f'{current_value}{current_mask}'

    on_set = set(on)

    merged = dfs[0].rename(columns={i: change_single(i, mask[0][1], mask[0][0], )
                                    for i in dfs[0].columns if i not in on_set})

    for single_df, single_mask in zip(dfs[1:], mask[1:]):
        single_df = single_df.rename(columns={i: change_single(i, single_mask[1], single_mask[0], )
                                              for i in single_df.columns if i not in on_set})
        merged = pd.merge(left=merged, right=single_df, how='inner', on=on)
    return merged


def train_pair(base_model, *, poisoning_func, poisoning_kwargs,
               X_train, y_train, X_test=None, y_test=None, verbose=False):
    X_train_poisoned, y_train_poisoned = poisoning_func(**poisoning_kwargs).fit(X_train, y_train).transform(
        X_train, y_train)
    target_model_clean = base_model().fit(X_train, y_train)
    target_model_poisoned = base_model().fit(X_train_poisoned, y_train_poisoned)

    if verbose and X_test is not None and y_test is not None:
        print(f'TARGET MODEL: clean:\taccuracy:'
              f'{metrics.accuracy_score(y_test, target_model_clean.predict(X_test))}')
        print(f'TARGET MODEL: poisoned:\taccuracy:'
              f'{metrics.accuracy_score(y_test, target_model_poisoned.predict(X_test))}')

    return target_model_clean, target_model_poisoned


@typing.runtime_checkable
class EstimatorProtocol(typing.Protocol):

    def fit(self, X, y, **fit_kwargs) -> "EstimatorProtocol":
        pass

    def predict(self, X):
        pass


SKLearnPipeline = typing.TypeVar('SKLearnPipeline', bound=pipeline.Pipeline)
SKLearnEstimator = typing.TypeVar('SKLearnEstimator', bound=base.BaseEstimator)

from sklearn import preprocessing


TAggregator = typing.Tuple[int, str,
typing.List[int],
typing.Tuple[
    typing.Callable[[typing.List[np.ndarray], ...], np.ndarray],
    typing.Callable[[typing.List[np.ndarray]], typing.Tuple[tuple, dict]]
]
]

TStep = typing.Tuple[str,]
import dataclasses

# T = typing.TypeVar('T')
VisitOutput = typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]


@dataclasses.dataclass
class StepVisitOutput:
    pass


#    np.ndarray], T]

# def VisitOutput(t):
#     return typing.Tuple[typing.Union[typing.Tuple[np.ndarray, np.ndarray], np.ndarray], t]

@typing.runtime_checkable
class SKLearnTransformerFitTransform(typing.Protocol):

    def fit_transform(self, *args, **kwargs):
        pass


class SKLearnTransformerFitAndTransform(typing.Protocol):

    def fit(self, X, y=None, **kwargs):
        pass

    def transform(self, X, y=None, **kwargs):
        pass


Transformer = typing.TypeVar('Transformer', bound=typing.Union[SKLearnTransformerFitTransform,
base.TransformerMixin, SKLearnTransformerFitAndTransform])


def check_admitted_value(got: T, admissible: typing.List[T]):
    if got not in admissible:
        raise ValueError(f'Unknown value: {got}, admissible: {admissible}')


def to_1d_or_raise(X):
    X_ = X
    if len(X_.shape) > 2:
        raise ValueError(f'X must be 1d or be converted to 1d, got shape: {X_.shape}')
    if len(X_.shape) == 2 and X_.shape[1] != 1:
        raise ValueError(f'X must be 1d or be converted to 1d, got shape: {X_.shape}')
    if len(X_.shape) == 2:
        # if X is not currently 1d, but it can be flattened (i.e., array([[1], [2]]), then we do it.
        X_ = np.reshape(X, (X_.shape[0],))
    return X_


def copy_if(arr, copy: bool):
    arr_ = arr
    if copy:
        arr_ = arr.copy()
    return arr_


def n_digits(highest: int) -> int:
    # not quite the most elegant approach but that's it.
    return len(str(highest))


def sample(X, y, sampling_for_training: float, rng):
    if 0 >= sampling_for_training or sampling_for_training > 1.0:
        raise ValueError(f'rate must be within (0, 1], got {sampling_for_training}')
    new_idx = rng.permutation(len(X))
    new_idx = new_idx[:int(np.round(len(X) * sampling_for_training))]
    return X[new_idx], y[new_idx]


class DistanceFromBoundaryWrapperMixin(ContainerRngAndJobsMixin, abc.ABC):
    # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

    def __init__(self, inner: typing.Optional[typing.Type[SKLearnEstimator]] = None,
                 inner_kwargs: typing.Optional[dict] = None, sampling_for_training: float = 1.0,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None, *args, **kwargs):
        # forward what we don't need.
        super().__init__(*args, **kwargs)
        # self.n_jobs = n_jobs
        # # bit of a copy, but otherwise we have the infamous diamond issue.
        # if rng is not None:
        #     self.rs = np.random.RandomState(rng)
        #     self.rng = np.random.default_rng(rng)
        # else:
        #     self.rs = None
        #     self.rng = np.random.default_rng()
        # self.seed = rng
        inner = inner if inner is not None else svm.LinearSVC
        inner_kwargs = inner_kwargs if inner_kwargs is not None else {}
        if 'SVC' in str(inner):
            if 'dual' not in inner_kwargs:
                inner_kwargs = {'dual': 'auto'}
        # removed inheritance from the estimator wrapper because it gives more issues than solutions.
        self.inner_ = multiclass.OneVsRestClassifier(**{
            'estimator': inner(**inner_kwargs if inner_kwargs is not None else {}),
            'n_jobs': n_jobs
        })
        self.sampling_for_training = sampling_for_training

    @abc.abstractmethod
    def _scale_input(self, X):
        pass

    def _fit(self, X, y=None, fit_kwargs=None, **kwargs):
        # if 0 >= self.sampling_for_training or self.sampling_for_training > 1:
        #     raise ValueError('sample_for_training must be 0<=x<=1')
        # # we randomly select a subset for training.
        # new_idx = self.rng.permutation(len(X))
        # new_idx = new_idx[:int(np.round(len(X) * self.sampling_for_training))]
        # X_ = self._scale_input(X[new_idx])
        # # train the one vs all classifier
        # self.inner_ = self.inner_.fit(X_, y[new_idx])
        X_, y_ = sample(X=X, y=y, sampling_for_training=self.sampling_for_training, rng=self.rng)
        X_ = self._scale_input(X_)
        # train the one vs all classifier
        self.inner_ = self.inner_.fit(X_, y_)
        return self


def filter_columns(columns: typing.Iterable[str],
                   patterns_to_exclude: typing.Optional[typing.Iterable[str]] = None,
                    patterns_to_include: typing.Optional[typing.Iterable[str]] = None,
                   filter_as_union: bool = True,
                   ) -> typing.List[str]:
    new_cols_after_excluded = []
    if patterns_to_exclude is None and patterns_to_include is None:
        return list(columns)
    if patterns_to_exclude is not None:
        for col in columns:
            found = False
            for excluded in patterns_to_exclude:
                if excluded in col:
                    found = True
            if not found:
                new_cols_after_excluded.append(col)
    new_cols_after_inclusion = []
    if patterns_to_include is None:
        new_cols_after_inclusion = new_cols_after_excluded
    else:
        starting_option = columns
        if not filter_as_union:
            starting_option = new_cols_after_excluded
        for col in starting_option:
        # for col in columns:
            for included in patterns_to_include:
                if included in col:
                    new_cols_after_inclusion.append(col)

    if filter_as_union:
        result = set(new_cols_after_excluded).union(new_cols_after_inclusion)
    else:
        result = new_cols_after_inclusion
        # add key perc points and so on.
        if const.KEY_PERC_DATA_POINTS in columns and const.KEY_PERC_DATA_POINTS not in result:
            result.append(const.KEY_PERC_DATA_POINTS)
        if const.KEY_PERC_FEATURES in columns and const.KEY_PERC_FEATURES not in result:
            result.append(const.KEY_PERC_FEATURES)

    return sorted(list(result))


def df_negate_filter(df: pd.DataFrame, patterns_to_exclude: typing.Optional[typing.List[str]] = None) -> pd.DataFrame:
    new_cols = filter_columns(df.columns, patterns_to_exclude)
    return df[new_cols]


def check_col_size(expected: int, got: typing.Optional[typing.Sequence[str]] = None):
    """
    Checks that the length of got equals expected if both are not None, raising ValueError.
    :param expected:
    :param got:
    :return:
    """
    if got is not None and len(got) != expected:
        raise ValueError(f'The length of columns is different than shape[1]: '
                         f'{len(got)} != {expected}')


def get_default_column_name(n_cols) -> typing.List[str]:
    """
    Giving `n_cols` it returns an array with default column names, that is `[f'col{i}' for i in range(n_cols)]`.
    :param n_cols:
    :return:
    """
    return [f'col{i}' for i in range(n_cols)]


def check_and_get_columns(expected: int, got: typing.Optional[typing.Sequence[str]] = None):
    """
    Checks that the length of `got` matches `expected` and returns a set of default column names
    for it.
    :param expected:
    :param got:
    :return:
    """
    if got is not None and len(got) != expected:
        raise ValueError(f'The length of columns is different than shape[1]: '
                         f'{len(got)} != {expected}')
    if got is None:
        return get_default_column_name(n_cols=expected)
    return got


def get_duplicates(all_names: typing.Sequence[str]) -> typing.List[str]:
    set_names = set(all_names)

    if len(set_names) != len(all_names):
        return [k for k, v in collections.Counter(all_names).items() if v > 1]
    return []


def get_pipeline_name_raw(full_name: str, short_name: str):
    if len(full_name) > 20 and short_name is not None:
        return short_name
    return full_name


def load_dataset_from_csv(dataset_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    # we need to read the columns manually
    # otherwise numpy creates array of np.void
    with open(dataset_path) as dataset_file:
        columns = dataset_file.readline().strip().split(',')

    array = np.genfromtxt(dataset_path, delimiter=',', skip_header=1)

    # col = columns[0]
    found = False
    col_idx = 0
    feature_mask = []
    for i, column in enumerate(columns):
        if column == const.COORD_LABEL:
            # col = column
            col_idx = i
            found = True
        else:
            feature_mask.append(i)

    if not found:
        raise ValueError(f'The dataset does not contain a column named \'{const.COORD_LABEL}\', '
                         f'I cannot proceed.')

    # now, we split the array extracting the labels.
    y = array[:, col_idx]

    X = array[:, feature_mask]

    return X, y


class EmptyModelException(Exception):
    pass