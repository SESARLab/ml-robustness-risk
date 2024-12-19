import numpy as np
import pytest
from sklearn import datasets, cluster

from . import composition_pre_post


@pytest.mark.parametrize('obj, expected_n_labels', [
    (
            composition_pre_post.GrouperNoGroup(),
            1
    ),
    (
            composition_pre_post.GrouperLabels(),
            2
    ),
    (
            composition_pre_post.GrouperClusteringAutomatic(clustering_kwargs={'min_cluster_size': 5}),
            None
    ),
    (
            composition_pre_post.GrouperClusteringAutomatic(),
            None
    ),
    (
            composition_pre_post.GrouperClusteringManual(clustering_kwargs={'n_clusters': 5}),
            5
    ),
])
def test_grouping_basic(obj: composition_pre_post.AbstractGrouper, expected_n_labels):
    X, y = datasets.make_classification()
    groups = obj.fit_transform(X, y)
    assert len(groups) == len(y)
    if expected_n_labels is not None:
        assert obj.n_groups == expected_n_labels
    if hasattr(obj, 'clustering') and hasattr(obj.clustering, 'labels_'):
        assert len(obj.clustering.labels_) == expected_n_labels


@pytest.mark.parametrize('obj', [
    composition_pre_post.AggregatorNoAggregator(),
    composition_pre_post.AggregatorAverage(),
    composition_pre_post.AggregatorPercentile(),
    composition_pre_post.AggregatorPercentile(percentile=[10, 90], how='weibull'),
])
def test_aggregator(obj):
    X, y = datasets.make_classification()
    result = obj.fit_transform(X, y)
    assert len(result) == len(X)
    assert np.all((result == 0) | (result == 1) | (result ==  X))
