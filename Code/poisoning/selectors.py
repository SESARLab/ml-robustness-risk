import abc
import enum
import typing
import warnings

# import shap
# from interpret import blackbox
import numpy as np
from scipy.spatial import distance
from sklearn import cluster, metrics, preprocessing

import utils
from . import base

T_SelectionInfo = typing.TypeVar('T_SelectionInfo', bound=base.AbstractSelectionInfo)


class AbstractSelector(abc.ABC, typing.Generic[T_SelectionInfo]):

    @property
    @abc.abstractmethod
    def selection_type(self) -> typing.Type:
        # unfortunately type assertions on generic is not possible, so we must
        # require to implement this method.
        pass

    def _check_type(self, got: typing.Optional[T_SelectionInfo] = None):
        if not isinstance(got, self.selection_type):
            raise ValueError(f'{self.__class__.__name__}: requires selection type '
                             f'{self.selection_type}, got: {type(got)}')

    @abc.abstractmethod
    def fit(self, X=None, y=None, selection_info: typing.Optional[T_SelectionInfo] = None) -> "AbstractSelector":
        pass

    @abc.abstractmethod
    def predict(self, X=None, y=None, selection_info: typing.Optional[T_SelectionInfo] = None
                ) -> typing.List[np.ndarray]:
        """
        Returns the numerical indices of the most suited data points to poison, the size depends on the selector.
        :param X:
        :param y:
        :param selection_info:
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def output_size() -> int:
        pass


class SelectorRandom(utils.ContainerRngAndJobsMixin, AbstractSelector):
    """
    Performs a totally random selection, ignoring even the labels. As such, it does not provide any guarantees.

    It is equivalent to the following code.
    ----
    >>> from numpy import random
    >>> X = ...
    >>> indices = np.arange(len(X))
    >>> random.default_rng().shuffle(indices)
    """

    @property
    def selection_type(self) -> typing.Type:
        return typing.Optional[base.SelectionInfoEmpty]

    @staticmethod
    def output_size() -> int:
        return 1

    def __init__(self, rng: typing.Optional[int] = None,
                 n_jobs: typing.Optional[int] = None, **kwargs):
        super().__init__(rng=rng, n_jobs=n_jobs, **kwargs)

    def fit(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None):
        self._check_type(got=selection_info)
        return self

    def predict(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None):
        self._check_type(got=selection_info)
        if X is None:
            raise ValueError('X must be not None.')
        indices = np.arange(len(X))
        self.rng.shuffle(indices)
        return [indices]

    def __eq__(self, other):
        if not isinstance(other, SelectorRandom):
            return False
        return True


class SelectorClustering(utils.EstimatorWrapperMixin, AbstractSelector):
    """
    Examples
    ------
    >>> from sklearn import datasets
    >>>
    >>> X, y = datasets.make_classification(100, 10)
    >>> sel  = SelectorClustering().fit(X, y, from_label=1, to_label=0)
    >>> idx = sel.predict(X, y, from_label=1, to_label=0)
    >>> idx
    np.array([90, 75, 45, ..., 1])
    >>> of_from_from_label = np.argwhere(y == 1).flatten()
    np.array([1, ..., 45, 75, 90])
    >>> len(np.intersect1d(of_from_from_label, idx)) == len(of_from_from_label)
    True
    """

    @staticmethod
    def output_size() -> int:
        return 1 #2

    @property
    def selection_type(self) -> typing.Type:
        return typing.Optional[base.SelectionInfoEmpty]

    def __init__(self, n_clusters: int = 2,
                 distance_exp: typing.Optional[int] = np.inf,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None,
                 inner_algo_clazz: typing.Optional[typing.Type[utils.SKLearnEstimator]] = cluster.KMeans,
                 inner_kwargs: typing.Optional[dict] = None,
                 scaling: typing.Optional[utils.Transformer] = None,
                 **kwargs):
        inner_kwargs = inner_kwargs if inner_kwargs is not None else {}
        inner_kwargs['n_clusters'] = n_clusters if n_clusters is not None else 2
        inner_kwargs['n_init'] = 'auto'
        # inner_kwargs['n_jobs'] = n_jobs
        inner_kwargs['random_state'] = rng
        super().__init__(inner=inner_algo_clazz, inner_kwargs=inner_kwargs, rng=rng, n_jobs=n_jobs, **kwargs)
        self.distance_exp = distance_exp if distance_exp is not None else 2
        self.cluster_to_real_ = None
        self.real_to_cluster_ = None
        self.scaling = scaling if scaling is not None else preprocessing.MinMaxScaler()

    def __eq__(self, other):
        if not isinstance(other, SelectorClustering):
            return False
        if self.distance_exp != other.distance_exp:
            return False
        # if type(self.inner) != type(other.inner):
        if not isinstance(self.inner, other.inner.__class__):
            return False
        return True

    def fit(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None):
        self._check_type(got=selection_info)
        if X is None or y is None:
            raise ValueError('X and y must be not None.')
        X_ = self.scaling.fit_transform(X)

        # must call explicitly because of multiple inheritance
        utils.EstimatorWrapperMixin.fit(self, X_, y)
        # ok now kmeans is fitted. We need to map
        # our labels, with the labels assigned by kmeans.
        # estimated_labels = self.inner.predict(X)
        self.cluster_to_real_, self.real_to_cluster_ = utils.LabelMapper(
            cluster_centers=self.inner.cluster_centers_).fit(X=X_).transform(X=X_, y=y)
        return self

    def predict(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None):
        self._check_type(got=selection_info)
        if X is None or y is None or selection_info is None:
            raise ValueError('X,y, and selection_info must be not None.')
        X_ = self.scaling.fit_transform(X)

        # from_label_ = self.real_to_cluster_[int(selection_info.from_label)]
        # to_label_ = self.real_to_cluster_[int(selection_info.to_label)]
        # idx = self._sort(X=X_,
        #                  y=y,
        #                  from_label=from_label_,
        #                  to_label=to_label_)
        assert selection_info is not None
        result = self._sort(X_, y)
        return result

    def _sort(self, X, y):
        # # retrieve the target centroid.
        # target = self.inner.cluster_centers_[self.real_to_cluster_[selection_info.from_label]]
        # distances = distance.cdist(X, target.reshape(1, -1), 'minkowski',
        #                                    p=float(self.distance_exp))
        # # then we need to reshape distances back to a 1d array
        # current_distances = distances.flatten()

        # The problem we have is that we cannot do argsort on 2 sub-arrays: when we
        # "merge" them after argsort, those indices are actually incorrect because they refer
        # to the indices of the sub-array.
        # other_indices = np.where(y == label)[0]
        # so we create a "complete" array containing the highest distance possible,
        # so that it won't mess up with the sorting nor with the retrieved distances.
        full = np.repeat(np.inf, len(X))
        for label in np.unique(y).astype(int):
            # select the data points of *opposite* label.
            current_indices = np.where(y != label)[0]
            # retrieve the target centroid.
            target = self.inner.cluster_centers_[self.real_to_cluster_[label]]
            # computes distances of each data point from the target,
            # reshape to make target a 2d array with one row.
            current_distances = distance.cdist(X[current_indices], target.reshape(1, -1), 'minkowski',
                                       p=float(self.distance_exp))

            # then we need to reshape distances back to a 1d array
            current_distances = current_distances.flatten()


            full[current_indices] = current_distances
            # now we sort the indices.
            # sorted_idx = np.argsort(full)

            # the issue is here
            # all_sorted_idx[current_indices] = sorted_idx

            # y_permuted = y[idx]
            # result.append(idx[y_permuted == label])
        # return result
        # now we can finally do the argsort.
        sorted_idx = np.argsort(full)
        sorted_idx = np.flip(sorted_idx)

        # just because.
        if np.count_nonzero(full == np.inf) != 0:
            raise ValueError(f'Clustering did not converge. Got {np.count_nonzero(full == np.inf)} '
                             'data points equal to \'np.inf\'')

        return [sorted_idx]


class SelectorBoundary(utils.DistanceFromBoundaryWrapperMixin, AbstractSelector):

    def __init__(self, inner: typing.Optional[typing.Type[utils.SKLearnEstimator]] = None,
                 inner_kwargs: typing.Optional[dict] = None,
                 sampling_for_training: float = 1.0,
                 scaling: typing.Optional[utils.Transformer] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None,
                 *args, **kwargs):
        kwargs['inner'] = inner
        kwargs['inner_kwargs'] = inner_kwargs
        kwargs['sampling_for_training'] = sampling_for_training
        kwargs['rng'] = rng
        kwargs['n_jobs'] = n_jobs
        super().__init__(*args, **kwargs)
        self.scaling = scaling if scaling is not None else preprocessing.MinMaxScaler()

    @property
    def selection_type(self) -> typing.Type:
        return typing.Optional[base.SelectionInfoEmpty]

    def _scale_input(self, X):
        return self.scaling.fit_transform(X)

    def fit(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None
            ) -> "AbstractSelector":
        utils.DistanceFromBoundaryWrapperMixin._fit(self, X, y)
        return self

    def predict(self, X=None, y=None,
                selection_info: typing.Optional[base.SelectionInfoEmpty] = None
                ) -> typing.List[np.ndarray]:
        distances = self.inner_.decision_function(self._scale_input(X))
        idx = np.argsort(distances)
        return [idx]

    @staticmethod
    def output_size() -> int:
        return 1


class SelectorSCLFA(utils.EstimatorWrapperMixin, AbstractSelector):

    def __init__(self, n_clusters: int = 2,
                 inner_kwargs: typing.Optional[dict] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None,
                 sampling_for_training: float = 1.0,
                 scaling: typing.Optional[utils.Transformer] = None,
                 *args, **kwargs):
        inner_kwargs = inner_kwargs if inner_kwargs is not None else {}
        inner_kwargs['n_clusters'] = n_clusters if n_clusters is not None else 2
        inner_kwargs['n_init'] = 'auto'
        # inner_kwargs['n_jobs'] = n_jobs
        inner_kwargs['random_state'] = rng
        self.sampling_for_training = sampling_for_training
        self.silhouette_value_ = None
        super().__init__(inner=cluster.KMeans, inner_kwargs=inner_kwargs, rng=rng, n_jobs=n_jobs, *args, **kwargs)
        self.scaling = scaling if scaling is not None else preprocessing.MinMaxScaler()

    def _scale_input(self, X):
        return self.scaling.fit_transform(X)

    def fit(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None):
        self._check_type(got=selection_info)
        if X is None or y is None:
            raise ValueError('X and y must be not None.')

        X_, y_ = utils.sample(X=X, y=y, sampling_for_training=self.sampling_for_training, rng=self.rng)
        # must call explicitly because of multiple inheritance
        utils.EstimatorWrapperMixin.fit(self, self._scale_input(X_), y_)
        return self

    @property
    def selection_type(self) -> typing.Type:
        return typing.Optional[base.SelectionInfoEmpty]

    @staticmethod
    def output_size() -> int:
        return 1

    def predict(self, X=None, y=None, selection_info: typing.Optional[base.SelectionInfoEmpty] = None
                ) -> typing.List[np.ndarray]:
        # now according to the attack, we retrieve the silhouette value of each point.
        # use this in a genetic algorithm.
        self.silhouette_value_ = metrics.silhouette_samples(self._scale_input(X), self.inner.predict(X))
        # sort according to the silhouette value, the lower, the higher fitness for poisoning
        idx = np.argsort(self.silhouette_value_)
        return [idx]
