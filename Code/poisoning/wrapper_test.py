import numpy as np
import pytest
from sklearn import datasets

from . import base, performers, selectors, wrapper

X, y = datasets.make_classification(100, 50)


@pytest.mark.parametrize('poisoning', [
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorClustering(),
        performer=performers.PerformerLabelFlippingMonoDirectional(),
        perform_info=base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10)
    ),
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorClustering(),
        performer=performers.PerformerLabelFlippingMonoDirectional(),
        perform_info=base.PerformInfoMonoDirectional(from_label=0, to_label=1, perc_data_points=30)
    ),
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorRandom(),
        performer=performers.PerformerLabelFlippingBiDirectional(),
        perform_info=base.PerformInfoBiDirectionalMirrored(perc_data_points=10, label_a=0, label_b=1)
    ),
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorBoundary(),
        performer=performers.PerformerLabelFlippingMonoDirectional(),
        perform_info=base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10)
    ),
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorSCLFA(),
        performer=performers.PerformerLabelFlippingMonoDirectional(),
        perform_info=base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10)
    ),
    wrapper.Poisoning(
        selection_info=base.SelectionInfoEmpty(),
        selector=selectors.SelectorSCLFA(),
        performer=performers.PerformerLabelFlippingBiDirectional(),
        perform_info=base.PerformInfoBiDirectionalMirrored(label_a=0, label_b=1, perc_data_points=10)
    )
])
def test_wrapper(poisoning: wrapper.Poisoning):
    poisoning.fit(X, y)
    X_, y_ = poisoning.transform(X, y)

    assert len(X_) == len(X)
    assert len(y_) == len(y)

    idx_of_poisoned = poisoning.performer.idx_of_poisoned_data_points
    idx_of_non_poisoned = np.setdiff1d(np.arange(len(X)), idx_of_poisoned)

    if poisoning.performer.modified_parts() == base.ModifiedPartOfPoints.y:
        # unchanged on not-involved data points.
        assert np.all(y_[idx_of_non_poisoned] == y[idx_of_non_poisoned])
        assert np.all(y_[idx_of_poisoned] != y[idx_of_poisoned])

