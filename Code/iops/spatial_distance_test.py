import pytest
from sklearn import datasets, cluster, svm

from . import base, base_test, spatial_distance


X, y = datasets.make_classification()


@pytest.mark.parametrize('algo_func, kwargs, expected_shape', [
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_BOTH
                }
            },
            (len(X), 2)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_BOTH
                }
            },
            (len(X), 2)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_CURRENT_CLASS
                }
            },
            (len(X), 1)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_CURRENT_CLASS
                }
            },
            (len(X), 1)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_OTHER_CLASS
                }
            },
            (len(X), 1)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.CLUSTERING,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    'clustering_clazz': cluster.KMeans,
                    'direction': base.Direction.FROM_OTHER_CLASS
                }
            },
            (len(X), 1)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.BOUNDARY,
                'reverse_how': base.ReverseType.SUBTRACT_BY_MAX,
                'inner_kwargs': {
                    'inner': svm.LinearSVC,
                    'inner_kwargs': {'dual': 'auto'},
                    'n_jobs': -1,
                }
            },
            (len(X), 1)
    ),
    (
            spatial_distance.IoPDistance,
            {
                'how': spatial_distance.IoPDistanceType.BOUNDARY,
                'reverse_how': base.ReverseType.DIVIDE_BY_1,
                'inner_kwargs': {
                    'inner': svm.LinearSVC,
                    'inner_kwargs': {'dual': 'auto'},
                    'n_jobs': -1,
                }
            },
            (len(X), 1)
    )
])
def test_iop(algo_func, kwargs, expected_shape):
    base_test.help_test_iop(algo_func, kwargs=kwargs, expected_shape=expected_shape, X_=X, y_=y)