import enum
import typing

import numpy as np
from numpy import ma
from sklearn.utils import validation

from . import spatial_common
from . import base

class PositionerNeighborKDistance(spatial_common.SpatialColumnsNotObeyingMixin,
                                  spatial_common.AbstractPositionerWithDirectionMixin,
                                  spatial_common.AbstractNeighborPositioner,
                                  ):

    def __init__(self, direction: base.Direction, neighbor_kwargs: typing.Optional[typing.Dict] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None, *args, **kwargs):
        super().__init__(direction=direction, neighbor_kwargs=neighbor_kwargs, rng=rng, n_jobs=n_jobs, *args, **kwargs)

    @property
    def output_shape(self) -> typing.Tuple[int, ...]:
        return (2 if self.direction == base.Direction.FROM_BOTH else 1, )

    @property
    def returns_scaled_output(self) -> bool:
        return False

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

    def _apply(self, X, y, direction: base.DirectionInner):
        X_ = X
        y_ = y
        validation.check_X_y(X_, y_)

        result = np.zeros(len(X))

        # need to set the type otherwise assignments fails.
        # output = np.repeat(np.inf, repeats=y.shape[0])
        distances, neigh = self.neigh.kneighbors(X_, n_neighbors=self.neigh.n_neighbors)
        # distances is an array where distances.shape[0] == X.shape[0],
        # and distances.shape[1] == k or, if k is None, it considers the k specified in the creation.
        # Each cell is the distance.
        #
        # neigh has the same shape but instead of distances it contains
        # the indices of the corresponding neighbors in distances.
        #
        # The first column is however not useful.
        # In fact, the first column in neigh reports the same data point,
        # while distances, in turn, contains 0 (because it is the distance from itself),
        # so we need to remove it.
        distances = distances[:, 1:]
        neigh = neigh[:, 1:]

        # now this is interesting, but we are interested in the distances
        # from the opposite label only.
        # y[neigh] is a multidimensional array
        # e.g., given a query of 2 points with k=5, y[neigh] is
        # array([[1, 1, 1, 1, 0],
        #        [1, 0, 1, 1, 0]])
        labels = np.unique(y)
        for current_label in np.nditer(labels):
            # first, we work on data points
            # whose label equals the current one.
            current_indices = np.where(y == current_label)[0]
            if len(current_indices) > 0:
                # Now, grab the neighbors belonging to the same class if direction == From.FROM_CURRENT_CLASS
                # or to the other class if direction == From.FROM_OTHER_CLASS
                if direction == base.DirectionInner.FROM_OTHER_CLASS:
                    neigh_of_other_label = y[neigh[current_indices]] != current_label
                else:
                    neigh_of_other_label = y[neigh[current_indices]] == current_label
                # y[neigh] is a multidimensional array
                # e.g., given a query of 2 points with k=5, y[neigh] is
                # array([[1, 1, 1, 1, 0],
                # [1, 0, 1, 1, 0]])
                # it means that:
                # - the first data point has 4 neighbors of opposite class, 1 of the same
                # - the second data point has 3 neighbors of the opposite class, 2 of the same.
                # Basically, the values *at 1 are those to be used for distance calculation*.
                # We therefore created a masked array, with the inverted mask (because '1' in the mask means
                # 'ignore such item').
                masked = ma.masked_array(distances[current_indices], mask=np.logical_not(neigh_of_other_label))
                # now, we need to retrieve the average. It may happen that the neighbors of a data point are fully
                # masked, because they have a label different from the current condition.
                # So, let us first retrieve the average.
                avg_distances_masked = ma.average(masked, axis=1)
                # Now, we retrieve the average as a non-masked array. But we need to be careful with the masked
                # value. If we are in the current class and a row is fully masked out, it means that the data point
                # does not have neighbors of its own class, which is bad. So the fill value should actually the highest.
                # If we are in the opposite class (e.g., 0) and a row is fully masked out, it means that the data point
                # does not have neighbors of the 'other' class (e.g., 1), which is good. So the fill value is 0.
                # HOWEVER: in both cases, we are just retrieving the average. Hence, we always use the highest value,
                # the semantics will be fixed later.
                # if direction == Direction.FROM_CURRENT_CLASS:
                #     # replace with the highest. Here, as a simple heuristic, we take the largest distance.
                #     fill_value = ma.max(masked)
                # else:
                #     fill_value = 0
                fill_value = ma.max(masked)

                avg_distances_unmasked = ma.filled(avg_distances_masked, fill_value)
                result[current_indices] = avg_distances_unmasked

        return result

    # def _fit_transform(self, X, y, **kwargs):
    #     if self.direction == base.Direction.FROM_BOTH:
    #         from_current = self._apply(X, y, direction=base.DirectionInner.FROM_CURRENT_CLASS)
    #         from_other = self._apply(X, y, direction=base.DirectionInner.FROM_OTHER_CLASS)
    #         values = np.vstack([from_current, from_other]).T
    #     elif self.direction == base.Direction.FROM_CURRENT_CLASS:
    #         values = self._apply(X, y, direction=base.DirectionInner.FROM_CURRENT_CLASS)
    #     else:
    #         values = self._apply(X, y, direction=base.DirectionInner.FROM_OTHER_CLASS)
    #     return values


class NeighborPositionerKCount(spatial_common.AbstractNeighborPositioner):

    @property
    def output_shape(self) -> typing.Tuple[int, ...]:
        return (1, )

    @property
    def columns_not_obeying_to_semantics(self) -> typing.Sequence[int]:
        # a higher value means <many data points of the same class>, which is not risky, so we do
        # need to change the semantics.
        return [0]

    @property
    def returns_scaled_output(self) -> bool:
        return False

    def _fit_transform(self, X, y, **kwargs):
        neigh = self.neigh.kneighbors(X, n_neighbors=self.neigh.n_neighbors, return_distance=False)
        # neigh is an array of shape (X.shape[0], k),
        # each i-th row contains *the indices* in X of the k-nearest neighbor of the i-th data point.
        # Example: array([[ 0, 30, 77, 78, 43],
        # ... ]) means that the k-nearest neighbors of the first data point are those in positions 0, 30, 77, 78, 43
        # in X.
        #
        # The first column is however not useful because ot reports the same data point, so we remove it.
        neigh = neigh[:, 1:]

        # set to the highest value possible (the number of neighbors returned for each point).
        output = np.repeat(self.neigh.n_neighbors, repeats=len(X)).astype(float)

        # now the idea is to count how many elements of *SAME* class
        # are found in the neighborhood of each element.
        labels = np.unique(y)
        for current_label in np.nditer(labels):

            # the indices of the elements whose label matches the current label.
            current_indices = np.where(y == current_label)[0]

            # with this operation we obtain an array whose shape is (len(current_indices), k),
            # where each i-th row refers to the neighbors of the i-th element.
            # Each i-th row is a boolean array, s.t. array[i, j] contains True if the j-th neighbor (out of k)
            # of the i-th data point has the same label of the i-th data point.
            # for instance:
            # [
            #   [   True,   False,  True,   True ]
            #   [   False,  False,  False,  False ]
            # ]
            # means that the 0th data point has the 0, 2, 3 neighbors with same label while 1 opposite.
            # The 1st data point has no neighbors of same label.
            neigh_of_interest = y[neigh[current_indices]] == current_label
            output[current_indices] = np.count_nonzero(neigh_of_interest, axis=1)

        # then, we just normalize the count.
        return output/self.neigh.n_neighbors


class IoPNeighborType(enum.Enum):
    K_DISTANCE = 'K_DISTANCE'
    K_LABEL_COUNT = 'K_LABEL_COUNT'
    # K_LABEL_MAJORITY_FIXED_VALUE = 'K_LABEL_MAJORITY_FIXED_VALUE'


class IoPNeighbor(base.IoP):

    def __init__(self, how: IoPNeighborType, inner_kwargs: typing.Optional[dict] = None,
                 reverse_how: typing.Optional[base.ReverseType] = None,
                 rng: typing.Optional[int] = None, n_jobs: typing.Optional[int] = None):
        super().__init__(rng=rng, n_jobs=n_jobs, inner_kwargs=inner_kwargs,
                         reverse_how=reverse_how,
                         mapping={
                             IoPNeighborType.K_DISTANCE: PositionerNeighborKDistance,
                             IoPNeighborType.K_LABEL_COUNT: NeighborPositionerKCount,
                         }, how=how)
