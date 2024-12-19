import abc
import typing

import numpy as np
from sklearn import cluster


class AbstractGrouper(abc.ABC):

    @abc.abstractmethod
    def fit_transform(self, X, y, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def requires_scaled_input(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def n_groups(self) -> int:
        pass


class GrouperNoGroup(AbstractGrouper):

    def fit_transform(self, X, y, **kwargs):
        return np.zeros(X.shape[0]).astype(int)

    @property
    def requires_scaled_input(self) -> bool:
        return False

    @property
    def n_groups(self) -> int:
        return 1


class GrouperLabels(AbstractGrouper):

    def __init__(self):
        self.groups_ = 2

    @property
    def requires_scaled_input(self) -> bool:
        return False

    def fit_transform(self, X, y, **kwargs):
        self.groups_ = len(np.unique(y))
        return y

    @property
    def n_groups(self) -> int:
        return self.groups_


class GrouperClusteringWithLabels(AbstractGrouper, abc.ABC):

    def __init__(self, clustering_clazz, clustering_kwargs: typing.Optional[typing.Dict] = None):
        self.clustering_kwargs = clustering_kwargs
        self.clustering = clustering_clazz(**clustering_kwargs or {})

    def fit_transform(self, X, y, **kwargs):
        self.clustering = self.clustering.fit(X=X, y=y)
        return self.clustering.labels_

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.clustering.labels_))


class GrouperClusteringAutomatic(GrouperClusteringWithLabels):

    def __init__(self, clustering_kwargs: typing.Optional[typing.Dict] = None):
        clustering_kwargs = clustering_kwargs or {}
        clustering_kwargs['copy'] = True
        super().__init__(clustering_clazz=cluster.HDBSCAN, clustering_kwargs=clustering_kwargs)

    @property
    def requires_scaled_input(self) -> bool:
        return True

class GrouperClusteringManual(GrouperClusteringWithLabels):

    def __init__(self, clustering_kwargs: typing.Optional[typing.Dict]):
        clustering_kwargs = clustering_kwargs or {}
        # clustering_kwargs['copy'] = True
        clustering_kwargs['compute_labels'] = True
        super().__init__(clustering_clazz=cluster.MiniBatchKMeans, clustering_kwargs=clustering_kwargs)

    @property
    def requires_scaled_input(self) -> bool:
        return True

class AbstractAggregator(abc.ABC):

    @abc.abstractmethod
    def fit_transform(self, X, y, **kwargs):
        pass

    # @property
    # @abc.abstractmethod
    # def requires_scaled_input(self) -> bool:
    #     pass

class AggregatorNoAggregator(AbstractAggregator):

    def fit_transform(self, X, y, **kwargs):
        return X

    # def requires_scaled_input(self) -> bool:
    #     return False

class AggregatorAverage(AbstractAggregator):

    def fit_transform(self, X, y, **kwargs):
        # axis=0 is safe if X is 1- or 2d.
        avg = np.average(X, axis=0)

        return np.where(X <= avg, 0, 1)


class AggregatorPercentile(AbstractAggregator):

    def __init__(self, percentile: typing.Optional[typing.List[int]] = None, how: str = 'linear'):
        self.percentiles = np.array(percentile) if percentile is not None else np.array([15, 85])
        self.how = how

    def _apply_to_1d(self, X_):
        results = np.percentile(X_, self.percentiles, method=self.how)
        return np.where(( X_ <= results[0]) | (X_ >= results[1]), 1, 0)

    def fit_transform(self, X, y, **kwargs):
        if len(self.percentiles) != 2:
            raise ValueError('percentile must have length 2')

        print(X.shape)

        # here we must differentiate whether the array is 1- or 2d.
        if len(X.shape) == 1 or X.shape[1] == 1:
            # if 1d, all good. (Note that this applies to array with shape (..., 1) as well.
            result = self._apply_to_1d(X)
        else:
            # apply separately column-wise.
            res_0 = self._apply_to_1d(X[:, 0])
            res_1 = self._apply_to_1d(X[:, 1])
            # and stack the results to the correct shape.
            result = np.vstack([res_0, res_1]).T

        return result
