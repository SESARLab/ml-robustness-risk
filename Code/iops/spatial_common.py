import abc
import typing

import numpy as np
from sklearn import neighbors, base

import utils
from . import base


class AbstractPositioner(base.IoPInner, abc.ABC):

    def __init__(self, # scaler_clazz: typing.Optional[typing.Type[utils.Transformer]] = preprocessing.MinMaxScaler,
                 # scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass


class SpatialColumnsNotObeyingMixin:

    @staticmethod
    def _columns_not_obeying_to_semantics(direction: base.Direction) -> typing.Sequence[int]:
        if direction == base.Direction.FROM_CURRENT_CLASS:
            # higher values in <distance-from-my-own-class> is risky, so we do not need to change the semantics
            return []
        elif direction == base.Direction.FROM_OTHER_CLASS:
            # higher values in <distance-from-my-own-class> is not risky, so we need to change the semantics
            return [0]
        else:
            # 0 is SAME CLASS, 1 is OTHER
            return [1]


class AbstractNeighborPositioner(utils.ContainerRngAndJobsMixin, AbstractPositioner, abc.ABC):

    def __init__(self, neighbor_kwargs: typing.Optional[typing.Dict] = None,
                 # scaler_clazz: typing.Optional[typing.Type[utils.Transformer]] = preprocessing.MinMaxScaler,
                 # scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None, *args, **kwargs):
        super().__init__(rng=rng, n_jobs=n_jobs, # scaler_clazz=scaler_clazz, scaler_kwargs=scaler_kwargs,
                         *args, **kwargs)
        self.neighbor_kwargs = neighbor_kwargs or {}
        self.neigh: neighbors.KNeighborsTransformer = None
        self.k = self.neighbor_kwargs.get('n_neighbors', 5)

    @abc.abstractmethod
    def _fit_transform(self, X, y):
        pass

    def fit_transform(self, X, y):
        # if we apply some grouping before, we need to dynamically adjust the size of k and re-initialize everything.
        if len(X) < self.k:
            self.k = len(X)
            self.neighbor_kwargs['n_neighbors'] = self.k

        # now, we can create the object using a safe value of k.
        self.neigh = neighbors.KNeighborsTransformer(**self.neighbor_kwargs)
        self.k = self.neigh.n_neighbors

        self.neigh = self.neigh.fit(X, y)
        return self._fit_transform(X, y)

    @property
    def requires_scaled_input(self) -> bool:
        # since we are working with knn.
        return True


class AbstractPositionerWithDirectionMixin(abc.ABC):

    def __init__(self,  direction: base.Direction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction

    def _fit_transform(self, X, y, **kwargs):
        if self.direction == base.Direction.FROM_BOTH:
            from_current = self._apply(X, y, direction=base.DirectionInner.FROM_CURRENT_CLASS)
            from_other = self._apply(X, y, direction=base.DirectionInner.FROM_OTHER_CLASS)
            values = np.vstack([from_current, from_other]).T
        elif self.direction == base.Direction.FROM_CURRENT_CLASS:
            values = self._apply(X, y, direction=base.DirectionInner.FROM_CURRENT_CLASS)
        else:
            values = self._apply(X, y, direction=base.DirectionInner.FROM_OTHER_CLASS)
        return values

    @abc.abstractmethod
    def _apply(self, X, y, direction):
        pass
