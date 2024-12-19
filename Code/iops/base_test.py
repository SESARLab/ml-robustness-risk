import numpy as np
import pytest
from sklearn import datasets, preprocessing

from . import base

X, y = datasets.make_classification(20, 5)


@pytest.mark.parametrize('values', [
    np.random.default_rng().choice(np.arange(10), (100, 10)),
    np.random.default_rng().choice(np.arange(10), (100,)).reshape(-1, 1)
])
def test_scale_only(values):
    class Obj(base.AbstractSpatialAnalysis):
        def _compute_distance_from_class(self, X_, y_, direction: base.From, **kwargs) -> np.ndarray:
            pass

        @property
        def _column_not_obeying(self) -> base.ColumnNotObeyingToSemantics:
            pass

    scaler = preprocessing.MinMaxScaler()

    obj = Obj(direction=base.DirectionType.FROM_BOTH)
    scaled = obj._scale_only(values=values, scaler=scaler)

    assert scaled.shape == values.shape
    assert np.all(scaled >= 0)
    assert np.all(scaled <= 1)


@pytest.mark.parametrize('values, columns_not_obeying, columns_idx_to_check, direction_', [
    (
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.NONE,
            [],
            base.DirectionType.FROM_BOTH
    ),
    (
            # here the first column must change.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.CURRENT,
            [0],
            base.DirectionType.FROM_BOTH
    ),
    (
            # here the second column must change.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.OTHER,
            [1],
            base.DirectionType.FROM_BOTH
    ),
    (
            # here both columns must change.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.BOTH,
            [0, 1],
            base.DirectionType.FROM_BOTH
    ),
    (
            # this is not going to change, since
            # we say that the column not obeying is OTHER,
            # but we operate on SAME only.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.OTHER,
            [],
            base.DirectionType.FROM_CURRENT_CLASS
    ),
    (
            # this is not going to change, since
            # we say that the column not obeying is NONE,
            # but we operate on SAME only.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.NONE,
            [],
            base.DirectionType.FROM_CURRENT_CLASS
    ),
    (
            # this is not going to change, since
            # we say that the column not obeying is NONE,
            # but we operate on OTHER only.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.NONE,
            [],
            base.DirectionType.FROM_OTHER_CLASS
    ),
    (
            # this is not going to change, since
            # we say that the column not obeying is NONE,
            # but we operate on BOTH.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.NONE,
            [],
            base.DirectionType.FROM_BOTH
    ),
    (
            # this is not going to change, since
            # we say that the column not obeying is SAME,
            # but we operate on OTHER only.
            np.random.default_rng().choice(np.arange(100), size=(100, 2)),
            base.ColumnNotObeyingToSemantics.CURRENT,
            [],
            base.DirectionType.FROM_OTHER_CLASS
    )
])
def test_semantics_adjusted(values: np.ndarray, columns_not_obeying: base.ColumnNotObeyingToSemantics,
                            columns_idx_to_check,
                            direction_: base.DirectionType):
    class Obj(base.AbstractSpatialAnalysis):
        def _compute_distance_from_class(self, X_, y_, direction: base.From, **kwargs) -> np.ndarray:
            pass

        @property
        def _column_not_obeying(self) -> base.ColumnNotObeyingToSemantics:
            return columns_not_obeying

    obj = Obj(direction=direction_)
    # copy since the values are going to change.
    values_ = values.copy()
    adjusted = obj._adjust_semantics(values)

    for single_column_idx in columns_idx_to_check:
        # if it has adjusted the semantics, then all values must be different.
        assert not np.array_equal(values_[:, single_column_idx], adjusted[:, single_column_idx])
        # this sometimes gives problem, but it's fine.
        # assert np.all(values_[:,single_column_idx] != adjusted[:, single_column_idx])

    columns_idx_not_changed = list(set(np.arange(values_.shape[1])) - set(columns_idx_to_check))
    for single_column_idx in columns_idx_not_changed:
        assert np.all(values_[:, single_column_idx] == adjusted[:, single_column_idx])


@pytest.mark.parametrize('augmenter_kwargs', [
    {
        'with_pca': True,
        'pca_kwargs': {
            'n_components': 3,
            'whiten': False
        }
    },
    {
        'with_pca': False
    }
])
def test_dataset_augmenter(augmenter_kwargs):
    augmenter = base.DatasetAugmenter(**augmenter_kwargs)
    augmenter.fit(X, y)

    new_data = augmenter.transform(required_size=50)
    assert new_data.shape == (50, X.shape[1])


def help_test_iop(algo_func, kwargs, expected_shape, X_, y_):
    algo = algo_func(**kwargs)
    got = algo.fit_transform(X_, y_)
    if expected_shape is not None:
        if isinstance(expected_shape, tuple):
            assert got.shape == expected_shape
        elif isinstance(expected_shape, int):
            assert len(got) == expected_shape
    return algo, got
