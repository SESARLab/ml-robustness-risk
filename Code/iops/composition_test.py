import enum

import numpy as np
import pytest
from sklearn import datasets

from . import base, composition, composition_pre_post, spatial_neigh


X, y = datasets.make_classification()

def test_composition_default():
    """
    We test that defaults are set correctly, and that the whole process works.
    :return:
    """
    composed = composition.IoPComposer(iop_clazz=spatial_neigh.IoPNeighbor,
                                       iop_kwargs={
                                           'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                                           'inner_kwargs': {
                                               'neighbor_kwargs': {
                                                   'n_neighbors': 3,
                                               }
                                           },
                                           'reverse_how': base.ReverseType.DIVIDE_BY_1,
                                       })

    result = composed.fit_transform(X, y)

    assert composed.grouper_clazz == composition_pre_post.GrouperNoGroup
    assert composed.aggregator_clazz == composition_pre_post.AggregatorNoAggregator
    assert composed.grouper.n_groups == 1

    assert result.shape == (X.shape[0], 1)


@pytest.mark.parametrize('composed, expected_shape', [
    (
            # one same as above but without empty kwargs
            composition.IoPComposer(iop_clazz=spatial_neigh.IoPNeighbor,
                                    iop_kwargs={
                                        'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                                        'inner_kwargs': {
                                            'neighbor_kwargs': {
                                                'n_neighbors': 3,
                                            }
                                        },
                                        'reverse_how': base.ReverseType.DIVIDE_BY_1,
                                    },
                                    grouper_clazz=composition_pre_post.GrouperClusteringManual,
                                    grouper_kwargs={'clustering_kwargs':{ 'n_clusters': 2}},
                                    aggregator_clazz=composition_pre_post.AggregatorAverage,
                                    ),
            (len(X), 1)
    ),
    (
            # one same as above but without empty kwargs
            composition.IoPComposer(iop_clazz=spatial_neigh.IoPNeighbor,
                                    iop_kwargs={
                                        'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                                        'inner_kwargs': {
                                            'neighbor_kwargs': {
                                                'n_neighbors': 3,
                                            }
                                        },
                                        'reverse_how': base.ReverseType.DIVIDE_BY_1,
                                    },
                                    grouper_clazz=composition_pre_post.GrouperClusteringManual,
                                    grouper_kwargs={'clustering_kwargs':{ 'n_clusters': 2}},
                                    aggregator_clazz=composition_pre_post.AggregatorPercentile,
                                    aggregator_kwargs={'percentile': [10, 90]}
                                    ),
            (len(X), 1)
    ),
    (
            # one same as above but without empty kwargs
            composition.IoPComposer(iop_clazz=spatial_neigh.IoPNeighbor,
                                    iop_kwargs={
                                        'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                                        'inner_kwargs': {
                                            'neighbor_kwargs': {
                                                'n_neighbors': 3,
                                            }
                                        },
                                        'reverse_how': base.ReverseType.DIVIDE_BY_1,
                                    },
                                    grouper_clazz=composition_pre_post.GrouperClusteringAutomatic,
                                    aggregator_clazz=composition_pre_post.AggregatorAverage,
                                    ),
            (len(X), 1)
    ),
    (
            # one same as above but without empty kwargs
            composition.IoPComposer(iop_clazz=spatial_neigh.IoPNeighbor,
                                    iop_kwargs={
                                        'how': spatial_neigh.IoPNeighborType.K_LABEL_COUNT,
                                        'inner_kwargs': {
                                            'neighbor_kwargs': {
                                                'n_neighbors': 3,
                                            }
                                        },
                                        'reverse_how': base.ReverseType.DIVIDE_BY_1,
                                    },
                                    grouper_clazz=composition_pre_post.GrouperClusteringAutomatic,
                                    grouper_kwargs={'clustering_kwargs':{ 'min_cluster_size': 5}},
                                    aggregator_clazz=composition_pre_post.AggregatorPercentile,
                                    aggregator_kwargs={'percentile': [10, 90]}
                                    ),
            (len(X), 1)
    ),
])
def test_composition(composed: composition.IoPComposer, expected_shape):
    result = composed.fit_transform(X, y)
    assert result.shape == expected_shape
