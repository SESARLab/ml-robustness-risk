import abc
import enum
import typing

import numpy as np
from sklearn import decomposition, preprocessing, model_selection, neighbors

import utils


class From(enum.Enum):
    FROM_CURRENT_CLASS = 'from_current'
    FROM_OTHER_CLASS = 'from_other'


class DirectionType(enum.Enum):
    FROM_CURRENT_CLASS = 'from_current'
    FROM_OTHER_CLASS = 'from_other'
    FROM_BOTH = 'from_both'

    def to_direction(self) -> From:
        assert self != DirectionType.FROM_BOTH
        if self == DirectionType.FROM_OTHER_CLASS:
            return From.FROM_OTHER_CLASS
        else:
            return From.FROM_CURRENT_CLASS


class ColumnNotObeyingToSemantics(enum.Enum):
    CURRENT = 'current'
    OTHER = 'other'
    NONE = 'none'
    BOTH = 'both'


# class AbstractSpatialAnalysis(utils.ContainerRngAndJobs, abc.ABC):
class AbstractSpatialAnalysis(abc.ABC):
    """
    Abstract class to retrieve the distance of a data point wrt the other class.

    It includes a couple of methods that child classes shall implement.

    - _compute_distance_from_class(X) returns a 1d array whose length matches len(X).
    In case the argument direction equals From.CURRENT_CLASS, it returns, for each data point, a number
    specifying how far the data point is wrt to its own class. If it is From.OTHER_CLASS, it returns,
    for each data point, a number specifying how far the data point is wrt to the other class.

    According to parameter `direction` passed to the constructor, this method shall be called twice to
    build a 2d array containing the distance from current (first column) and other (second column) class for each point.
    However, there are cases where this is not necessary, notably DistanceFromOtherNeighborKCount, where the results
    are always symmetric. In this case, it is fine for the child class to restrict the behavior of
    `_compute_distance_from_class(X)` provided that method `_forbidden_direction` returns the forbidden direction.

    Note that `_compute_distance_from_class(X)` shall return an array preserving the semantics *the lower, the better*.
    If this is not the case, it shall implement method `_column_not_obeying(self)` specifying which distance
    (CURRENT, i.e., from current class; OTHER, i.e., from other class; NONE, i.e., semantics is always preserved;
    BOTH, i.e., semantics is never preserved). This class will then take care of modifying the
    distance array properly.

    Note that `_column_not_obeying(self)` does not provide an implementation by itself and must be always sub-classed.
    """

    def __init__(self,
                 used_direction: DirectionType,
                 scaler: typing.Optional[typing.Type[utils.Transformer]] = preprocessing.MinMaxScaler, *args, **kwargs):
        self.scaler_type = scaler
        self.used_direction = used_direction

    @abc.abstractmethod
    def _compute_distance_from_class(self, X, y, direction: From, **kwargs) -> np.ndarray:
        pass

    @property
    def _requires_scaling_input(self) -> bool:
        return True

    @property
    def _requires_scaling_output(self) -> bool:
        return True

    @property
    @abc.abstractmethod
    def _column_not_obeying(self) -> ColumnNotObeyingToSemantics:
        pass

    @property
    @abc.abstractmethod
    def allowed_directions(self) -> typing.List[DirectionType]:
        pass

    @property
    def _require_scaler(self) -> bool:
        return True

    def check_fit(self,):
        if self.used_direction not in self.allowed_directions:
            raise ValueError(f'The chosen direction {self.used_direction} is not allowed')

    def _scale_input(self, X: np.ndarray):
        if self._requires_scaling_input:
            # axis = 0 means scaling columns independently (implicit).
            return self.scaler_type().fit_transform(X)
        return X

    def _scale_output(self, X: np.ndarray):
        # scaling output is a bit trickier because
        # the output may be 1d and so it requires reshaping.
        X_ = X
        if self._requires_scaling_output:
            if self.used_direction != DirectionType.FROM_BOTH:
                X_ = X_.reshape(-1, 1)
            return self.scaler_type().fit_transform(X_)
        return X_

    def _adjust_semantics(self, X: np.ndarray) -> np.ndarray:
        # first, find the cases where we do not need to adjust, that is
        # - where we operate on single column, and the column is not the one not obeying (i.e., when we operate on the
        # correct column only)
        # - where we operate on both columns, and they are both correct.
        column_idx = []
        if (self._column_not_obeying == ColumnNotObeyingToSemantics.NONE) or (
                self.used_direction == DirectionType.FROM_OTHER_CLASS and self._column_not_obeying
                != ColumnNotObeyingToSemantics.OTHER and self._column_not_obeying != ColumnNotObeyingToSemantics.BOTH) \
                or (self.used_direction == DirectionType.FROM_CURRENT_CLASS and self._column_not_obeying !=
                    self._column_not_obeying.CURRENT and self._column_not_obeying != ColumnNotObeyingToSemantics.BOTH):
            return X

        # otherwise, let's manage the cases extracting the index of the columns to fix.
        if self.used_direction == DirectionType.FROM_BOTH:
            if self._column_not_obeying == ColumnNotObeyingToSemantics.BOTH:
                column_idx = [0, 1]
            elif self._column_not_obeying == ColumnNotObeyingToSemantics.CURRENT:
                column_idx = [0]
            elif self._column_not_obeying == ColumnNotObeyingToSemantics.OTHER:
                column_idx = [1]
        else:
            # if working in one direction only, then the index is always 0.
            column_idx = [0]

        for idx in column_idx:
            max_from_current = np.max(X[:, idx])
            X[:, idx] = max_from_current - X[:, idx]
        return X

    def predict_proba(self, X, y, **kwargs):
        X_ = self._scale_input(X)

        if self.used_direction == DirectionType.FROM_BOTH:
            from_current = self._compute_distance_from_class(X_, y, direction=From.FROM_CURRENT_CLASS, **kwargs)
            from_other = self._compute_distance_from_class(X_, y, direction=From.FROM_OTHER_CLASS, **kwargs)
            # so that we return an array whose shape[0] matches X.shape[0] composed
            # of two features: distances_from_current, distances_from_other
            values = np.vstack([from_current, from_other]).T
        else:
            values = self._compute_distance_from_class(X_, y, direction=self.used_direction.to_direction(), **kwargs)

        # if scaler is not None:
        #     #  values = self.scale_with_scaler(values, scaler)
        #     values = self._scale_only(values, scaler)
        values = self._scale_output(values)

        values = self._adjust_semantics(values)

        if self.used_direction != DirectionType.FROM_BOTH:
            # need to ensure it returns an array of 1d array., i.e., ensure it still
            # returns a multidimensional array as in case FROM_BOTH.
            values = values.reshape(-1, 1)
        return values

    def predict(self, X, y, **kwargs):
        # if kwargs is None:
        #     kwargs = {}
        # if 'scaler' not in kwargs:
        #     kwargs['scaler'] = preprocessing.MinMaxScaler()
        return self.predict_proba(X, y, **kwargs)

    def transform(self, X, y, **kwargs):
        return self.predict(X, y, **kwargs)


class DatasetAugmenter:

    def __init__(self, with_pca: bool = True, pca_kwargs: typing.Optional[dict] = None,
                 kde_cv_kwargs: typing.Optional[dict] = None):
        self.with_pca = with_pca
        self.pca_kwargs = pca_kwargs if pca_kwargs is not None else {'n_components': 15, 'whiten': False}
        self.kde_cv_kwargs = kde_cv_kwargs if kde_cv_kwargs is not None else {'bandwidth': np.logspace(-1, 1, 20)}
        self.pca_: typing.Optional[decomposition.PCA] = None
        self.kde_: neighbors.KernelDensity = None

    def fit(self, X, y=None):
        X_ = X
        if self.with_pca:
            self.pca_ = decomposition.PCA(**self.pca_kwargs)
            X_ = self.pca_.fit_transform(X_)

        # use grid search cross-validation to optimize the bandwidth
        grid = model_selection.GridSearchCV(neighbors.KernelDensity(), self.kde_cv_kwargs)
        grid.fit(X_)

        # use the best estimator to compute the kernel density estimate
        self.kde_ = grid.best_estimator_

    def transform(self, X=None, y=None, required_size: typing.Optional[int] = None):

        if X is None and required_size is None:
            raise ValueError('X and required_size cannot be both None at the same time.')

        required_size = required_size if required_size is not None else len(X)

        new_data = self.kde_.sample(required_size)

        if self.with_pca:
            new_data = self.pca_.inverse_transform(new_data)

        return new_data


T = typing.TypeVar('T')


T_Enum = typing.TypeVar('T_Enum', bound=enum.Enum)


class IoPInner(abc.ABC):

    @abc.abstractmethod
    def fit_transform(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def returns_scaled_output(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def requires_scaled_input(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def columns_not_obeying_to_semantics(self) -> typing.Sequence[int]:
        pass

    @property
    @abc.abstractmethod
    def output_shape(self) -> typing.Tuple[int, ...]:
        """
        Specify the output shape. Note that the first item is input-dependent
        and cannot be known in advance. By convention

        - we do not report it the first *dimension*
        - even when it returns a flat array, we use `(1, )` (i.e., it will return something like `(100, 1)`.

        :return: output shape of all but the first dimension
        """
        # necessary for correct (and faster) allocation of the result
        # when used in composition
        pass


T_IoPInner = typing.TypeVar('T_IoPInner', bound=IoPInner)

class ReverseType(enum.Enum):

    DIVIDE_BY_1 = 'DIVIDE_BY_1'
    SUBTRACT_BY_MAX = 'SUBTRACT_BY_MAX'

    def reverse(self, a: np.ndarray):
        if self == ReverseType.DIVIDE_BY_1:
            # taken from: https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
            # with np.errstate(divide='ignore', invalid='ignore'):
            with np.errstate(all='raise'):
                # fill = 0
                # result = 1 / a
                # if np.isscalar(result):
                #     return result if np.isfinite(result) else fill
                # else:
                #     result[~np.isfinite(result)] = fill
                #     return result
                # when a is 0, it results in a division by 0, which is not good.
                # so we replace with eps.
                # return 1/np.where(a == 0, np.finfo(float).eps, a)
                # set 1 as result when a==0.
                result = np.divide(1, a, where=a!=0, out=np.ones(len(a)))
                # the issue with this approach is that the result is not scaled in [0, 1].
                # so we scale it.
                return preprocessing.minmax_scale(result)
            # return 1/a
        else:
            max_val = np.max(a)
            return max_val - a


class IoP(utils.ContainerRngAndJobsMixin, typing.Generic[T_Enum]):

    def __init__(self, how: T_Enum,
                 # admissible_types: typing.Type[T_Enum],
                 mapping: typing.Dict[T_Enum, typing.Type[T_IoPInner]],
                 inner_kwargs: typing.Optional[dict] = None,
                 scaler_clazz: typing.Optional[typing.Type[utils.Transformer]] = None,
                 scaler_kwargs: typing.Optional[typing.Dict] = None,
                 reverse_how: typing.Optional[ReverseType] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None):

        super().__init__(rng=rng, n_jobs=n_jobs)
        self.how = how
        self.inner_kwargs = inner_kwargs if inner_kwargs is not None else {}
        # self.admissible_types = [e for e in admissible_types]
        # self.admissible_types = [k for k in mapping.keys()]
        self.mapping = mapping

        self.scaler_clazz = scaler_clazz or preprocessing.MinMaxScaler
        self.scaler_kwargs = scaler_kwargs or {}

        self.reverse_how = reverse_how

        # NOTE: I know that it is against sklearn pattern to check it here,
        # but let's do it.
        if self.how not in self.mapping.keys():
            raise ValueError(f'Unknown \'how\'. Admissible values: {list(self.mapping.keys())}')

        step_func = self.mapping[self.how]

        args = {
            # TODO rng
            # 'rng': self.rng,
            'n_jobs': self.n_jobs
        }
        args |= self.inner_kwargs

        self.step_: IoPInner = step_func(**args)

        if len(self.step_.columns_not_obeying_to_semantics) > 0 and self.reverse_how is None:
            raise ValueError(f'This IoP {self.__class__} needs to reverse the semantics of its columns but no reversed is provided')

    def fit_transform(self, X, y):

        X_ = X
        # if self.step_.requires_scaled_input:
        #    scaler = self.scaler_clazz(**self.scaler_kwargs)
        #    X_ = scaler.fit_transform(X_)
        result = self.step_.fit_transform(X_, y)
        #if hasattr(self.step_, 'fit_transform'):
        #    result = self.step_.fit_transform(X_, y)
        #else:
        #    result = self.step_.fit(X_, y).transform(X_, y)

        # reshape the result if needed.
        if len(result.shape) == 1:
            result = result.reshape(-1, 1)

        if not self.step_.returns_scaled_output:
            # NOTE: this is a bit of an antipattern, why not using a pipeline?
            # According to the usage we made, it's easier this way.
            # scale.
            scaler = self.scaler_clazz(**self.scaler_kwargs)
            result = scaler.fit_transform(result)

        if len(self.step_.columns_not_obeying_to_semantics) > 0:
            for col in self.step_.columns_not_obeying_to_semantics:
                result[:, col] = self.reverse_how.reverse(result[:, col])

        return result


class DirectionInner(enum.Enum):
    """
    Used only internally.
    """
    FROM_CURRENT_CLASS = 'from_current'
    FROM_OTHER_CLASS = 'from_other'


class Direction(enum.Enum):
    FROM_CURRENT_CLASS = 'FROM_CURRENT_CLASS'
    FROM_OTHER_CLASS = 'FROM_OTHER_CLASS'
    FROM_BOTH = 'FROM_BOTH'


