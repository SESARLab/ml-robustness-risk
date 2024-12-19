import numpy as np
import pytest
from sklearn import datasets

from . import base, performers

X, y = datasets.make_classification(50, 50)


def _check(a, a_, expected_number_of_poisoned, idx_of_poisoned_data_points):
    # check that changes match
    diff_idx = np.argwhere(a != a_).flatten()
    assert len(diff_idx) == expected_number_of_poisoned
    # NOTE: we do not use np.all because points are sorted in a different order
    assert len(np.intersect1d(idx_of_poisoned_data_points, diff_idx)) == len(diff_idx)
    # and that other points did not change
    equal_idx = np.argwhere(a == a_).flatten()
    assert len(equal_idx) == len(a) - expected_number_of_poisoned
    assert np.all(a[equal_idx] == a_[equal_idx])


@pytest.mark.parametrize('performer, selected_idx, info', [
    (
            # situation 3: we return two arrays but only one is relevant.
            performers.PerformerLabelFlippingMonoDirectional(),
            [np.random.default_rng().permutation(np.argwhere(y == 0).flatten()).astype(int),
             np.random.default_rng().permutation(np.argwhere(y == 1).flatten()).astype(int)],
            base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10.0)
    ),
    (
            # situation 2: only the points whose label matches are returned.
            performers.PerformerLabelFlippingMonoDirectional(),
            [np.random.default_rng().permutation(np.argwhere(y == 1).flatten()).astype(int)],
            base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10.0)
    ),
    (
            # situation 3: all points are returned in an individual array.
            performers.PerformerLabelFlippingMonoDirectional(),
            [np.random.default_rng().permutation(len(X)).astype(int)],
            base.PerformInfoMonoDirectional(from_label=1, to_label=0, perc_data_points=10.0)
    ),
    (
            performers.PerformerLabelFlippingBiDirectional(),
            [np.random.default_rng().permutation(np.argwhere(y == 0).flatten()).astype(int),
             np.random.default_rng().permutation(np.argwhere(y == 1).flatten()).astype(int)],
            base.PerformInfoBiDirectionalMirrored(label_a=1, label_b=0, perc_data_points=10.0)
    ),
])
def test_performer(performer, selected_idx, info: base.AbstractPerformInfo):
    performer.fit(X=X, y=y, specific_args=info)
    X_, y_ = performer.transform(X=X, y=y, selected_idx=selected_idx, specific_args=info)

    expected_number_of_poisoned = info.get_number_of_data_points(len(X))

    assert len(X_) == len(X)
    assert len(y_) == len(y)

    assert expected_number_of_poisoned == len(performer.idx_of_poisoned_data_points)
    # lack of duplicates. Just in case.
    assert len(np.unique(performer.idx_of_poisoned_data_points)) == len(performer.idx_of_poisoned_data_points), print(
        f'{performer.idx_of_poisoned_data_points}')

    if performer.modified_parts() == base.ModifiedPartOfPoints.X:
        _check(X, X_, expected_number_of_poisoned=expected_number_of_poisoned,
               idx_of_poisoned_data_points=performer.idx_of_poisoned_data_points)
    elif performer.modified_parts() == base.ModifiedPartOfPoints.y:
        _check(y, y_, expected_number_of_poisoned=expected_number_of_poisoned,
               idx_of_poisoned_data_points=performer.idx_of_poisoned_data_points)
    else:
        _check(X, X_, expected_number_of_poisoned=expected_number_of_poisoned,
               idx_of_poisoned_data_points=performer.idx_of_poisoned_data_points)
        _check(y, y_, expected_number_of_poisoned=expected_number_of_poisoned,
               idx_of_poisoned_data_points=performer.idx_of_poisoned_data_points)
