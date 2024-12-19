import enum
import typing

import numpy as np
from numpy import linalg as la
from sklearn import cluster

from . import base
from . import spatial_common
import utils


class DistanceFromClusteringBinary(utils.ContainerRngAndJobsMixin,
                                   spatial_common.AbstractPositionerWithDirectionMixin,
                                   spatial_common.SpatialColumnsNotObeyingMixin):

    def __init__(self, direction: base.Direction, clustering_clazz: typing.Type,
                 clustering_kwargs: typing.Optional[typing.Dict] = None, rng: typing.Optional[int] = None,
                 n_jobs: typing.Optional[int] = None, distance_metric_exp: int = np.inf, *args, **kwargs) -> None:
        super().__init__(rng=rng, n_jobs=n_jobs, direction=direction, *args, **kwargs)
        self.clustering_clazz = clustering_clazz if clustering_clazz else cluster.KMeans
        self.clustering_kwargs = clustering_kwargs if clustering_kwargs else {}
        self.clustering_kwargs['n_clusters'] = 2
        self.clustering_kwargs['n_init'] = 'auto' # this avoids some warnings.
        self.cluster_ = self.clustering_clazz(**self.clustering_kwargs)
        self.distance_metric_exp = distance_metric_exp
        self.cluster_to_real: np.ndarray = None
        self.real_to_cluster: np.ndarray = None

    def fit_transform(self, X, y, *args, **kwargs):
        if not isinstance(self.cluster_, cluster.KMeans) and not isinstance(self.cluster_, cluster.MiniBatchKMeans):
            raise ValueError(f'Only kMeans and MiniBatchKMeans are allowed. Got: {type(self.cluster_)}')
        self.cluster_.fit(X, y, *args, **kwargs)
        self.cluster_to_real, self.real_to_cluster = utils.LabelMapper(
            cluster_centers=self.cluster_.cluster_centers_).fit(X=X).transform(X=X, y=y)
        return self._fit_transform(X, y, **kwargs)

    @property
    def output_shape(self) -> typing.Tuple[int, ...]:
        return (2 if self.direction == base.Direction.FROM_BOTH else 1, )

    @property
    def returns_scaled_output(self) -> bool:
        return False

    @property
    def requires_scaled_input(self) -> bool:
        return True

    @property
    def columns_not_obeying_to_semantics(self) -> typing.Sequence[int]:
        # if self.direction == base.Direction.FROM_CURRENT_CLASS:
        #     # higher values in <distance-from-my-own-class> is risky, so we do not need to change the semantics
        #     return []
        # elif self.direction == base.Direction.FROM_OTHER_CLASS:
        #     # higher values in <distance-from-my-own-class> is not risky, so we need to change the semantics
        #     return [0]
        # else:
        #     # 0 is SAME CLASS, 1 is OTHER
        #     return [1]
        return self._columns_not_obeying_to_semantics(direction=self.direction)

    def _apply(self, X, y, direction: base.Direction) -> np.ndarray:
        # here we just have to measure the distance of each entry in X
        # from their own centroid or from the centroid of the other class.
        output = np.repeat(np.inf, repeats=len(X))
        for current_label in np.unique(y).astype(int):
            target_cluster_center = self.cluster_.cluster_centers_[self.real_to_cluster[current_label]]
            if direction == base.Direction.FROM_OTHER_CLASS:
                # if we are retrieving the distance in modality From.FROM_OTHER_CLASS
                # then we are interested in elements different from current_label
                current_indices = np.where(y != current_label)[0]
            else:
                current_indices = np.where(y == current_label)[0]
            distances = la.norm(X[current_indices] - target_cluster_center, self.distance_metric_exp, axis=1)
            output[current_indices] = distances
        return output


class DistanceFromBoundary(utils.DistanceFromBoundaryWrapperMixin,
                           spatial_common.SpatialColumnsNotObeyingMixin,
                           spatial_common.AbstractPositionerWithDirectionMixin,
                           ):

    @property
    def output_shape(self) -> typing.Tuple[int, ...]:
        return (1, )

    def _scale_input(self, X):
        # scaling is performed out this IoP.
        return X

    def __init__(self, inner: typing.Optional[typing.Type[utils.SKLearnEstimator]] = None,
                 inner_kwargs: typing.Optional[dict] = None,
                 sampling_for_training: float = 1.0,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None,
                 *args, **kwargs):
        # inner = inner if inner is not None else svm.LinearSVC
        # if 'SVC' in str(inner):
        #     if inner_kwargs is None or 'dual' not in inner_kwargs:
        #         inner_kwargs = {'dual': 'auto'}
        # # removed inheritance from the estimator wrapper because it gives more issues than solutions.
        # self.inner_ = multiclass.OneVsRestClassifier(**{
        #     'estimator': inner(**inner_kwargs if inner_kwargs is not None else {}),
        #     'n_jobs': n_jobs
        # })
        kwargs['inner'] = inner
        kwargs['inner_kwargs'] = inner_kwargs
        kwargs['sampling_for_training'] = sampling_for_training
        super().__init__(direction=base.Direction.FROM_OTHER_CLASS, rng=rng, n_jobs=n_jobs, *args, **kwargs)
        # self.scaling = scaling if scaling is not None else preprocessing.MinMaxScaler()

    def fit_transform(self, X, y, *args, **kwargs):
        utils.DistanceFromBoundaryWrapperMixin._fit(self, X, y)
        return self._fit_transform(X, y, **kwargs)

    @property
    def returns_scaled_output(self) -> bool:
        return False

    @property
    def requires_scaled_input(self) -> bool:
        return True

    @property
    def columns_not_obeying_to_semantics(self) -> typing.Sequence[int]:
        return [0]

    def _apply(self, X, y, direction):
        # there is really nothing more to do. It returns the distance from the boundary.
        # One distance for each data point because we are in binary classification.
        # Must be wrapped by abs because the distance can be positive or negative, but we are
        # interested in the absolute value only.
        return np.abs(self.inner_.decision_function(X))

class IoPDistanceType(enum.Enum):
    CLUSTERING = 'CLUSTERING'
    BOUNDARY = 'BOUNDARY'


class IoPDistance(base.IoP):

    def __init__(self, how: IoPDistanceType = IoPDistanceType.CLUSTERING,
                 reverse_how: typing.Optional[base.ReverseType] = None,
                 inner_kwargs: typing.Optional[dict] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None
                 ):
        super().__init__(rng=rng, n_jobs=n_jobs, how=how, inner_kwargs=inner_kwargs,
                         reverse_how=reverse_how,
                         mapping={
                             IoPDistanceType.CLUSTERING: DistanceFromClusteringBinary,
                             IoPDistanceType.BOUNDARY: DistanceFromBoundary
                         })
