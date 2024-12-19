import pytest
from sklearn import datasets

from . import base, base_test, spatial_neigh


X, y = datasets.make_classification()


@pytest.mark.parametrize('obj, expected_shape', [
    (
        spatial_neigh.PositionerNeighborKDistance(direction=base.Direction.FROM_CURRENT_CLASS,
                                                 neighbor_kwargs={'n_neighbors': 5}),
        # NO: because it is computed on an individual class.
        # (len(X[y == 0]),)
        (len(X),)
    ),
    (
        spatial_neigh.PositionerNeighborKDistance(direction=base.Direction.FROM_CURRENT_CLASS),
        # (len(X[y == 0]),),
        (len(X),)
    ),
    (
        spatial_neigh.NeighborPositionerKCount(neighbor_kwargs={'n_neighbors': 5}),
        (len(X),)
    ),
    (
        spatial_neigh.NeighborPositionerKCount(),
        (len(X),)
    )
])
def test_inner_level(obj, expected_shape):
    result = obj.fit_transform(X, y)
    assert result.shape == expected_shape


@pytest.mark.parametrize('algo_func, kwargs, expected_shape', [
    (
        spatial_neigh.IoPNeighbor,
        {
            'how': spatial_neigh.IoPNeighborType.K_DISTANCE,
            'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
            'inner_kwargs': {
                'direction': base.Direction.FROM_BOTH,
                'neighbor_kwargs': {'n_neighbors': 6}
            }
        },
        (len(X), 2)
    ),
    (
            spatial_neigh.IoPNeighbor,
            {
                'how': spatial_neigh.IoPNeighborType.K_DISTANCE,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    'direction': base.Direction.FROM_BOTH,
                    'neighbor_kwargs': {'n_neighbors': 6}
                }
            },
            (len(X), 2)
    ),
    (
            spatial_neigh.IoPNeighbor,
            {
                'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
                'inner_kwargs': {
                    #'direction': spatial_neigh.DirectionOuter.FROM_BOTH,
                    'neighbor_kwargs': {'n_neighbors': 6}
                }
            },
            (len(X), 1) # because it is already reshaped
    ),
    (
            spatial_neigh.IoPNeighbor,
            {
                'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    # 'direction': spatial_neigh.DirectionOuter.FROM_BOTH,
                    'neighbor_kwargs': {'n_neighbors': 6}
                }
            },
            (len(X), 1)  # because it is already reshaped
    )
])
def test_iop(algo_func, kwargs, expected_shape):
    base_test.help_test_iop(algo_func, kwargs=kwargs, expected_shape=expected_shape, X_=X, y_=y)