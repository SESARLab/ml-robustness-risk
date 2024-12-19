import itertools
import typing

import numpy as np
from sklearn import datasets

from . import base

X_, y_ = datasets.make_classification(n_classes=2, n_samples=1000)


def _test_assignment(assignment_class, kwargs, integer_columns: typing.Optional[typing.Sequence[int]] = None):
    assign = assignment_class(**kwargs)
    _test_assignment_inner(assign=assign, N=kwargs['N'], integer_columns=integer_columns)


def _test_assignment_inner(assign, N, integer_columns: typing.Optional[typing.Sequence[int]] = None):
    X__ = X_
    # add some integer columns at the end if necessary.
    if integer_columns is not None:
        new_cols = np.random.default_rng().choice([0, 1], size=(len(X_), integer_columns))
        X__ = np.hstack([X__, new_cols])

    result = assign.fit_transform(X__, y_)
    if isinstance(result, tuple):
        result = result[0]
    assert len(result) == len(X_)
    # everything that is in the result is [0, N-1] and vice versa
    assert np.all(np.intersect1d(result, np.arange(N)) == np.arange(N))
    return result


def test_custom_quality_metrics():
    class SimpleAssignment(base.AbstractAssignmentSpread):

        @staticmethod
        def guarantee_even_fill() -> base.OutputGuarantee:
            return base.OutputGuarantee.ALWAYS

        @staticmethod
        def strategy() -> base.AssignmentStrategy:
            return base.AssignmentStrategy.DISTRIBUTE

        def __init__(self, N):
            super().__init__(N=N)

        def fit_transform(self, X, y=None):
            assignment = np.tile(np.arange(self.n_clusters), int(np.ceil(len(X_) / self.n_clusters))).astype(int)
            if len(assignment) > len(X_):
                assignment = base.trim_assignment(assignment, len_second=len(X_))
            self.fill_assigned(result=assignment)
            return assignment

    obj = SimpleAssignment(5)
    obj.fit_transform(X_, y_)

    # now let's use the metrics

    poisoning_idx = np.random.default_rng().choice([0, 1], len(X_))
    risk_idx = np.random.default_rng().choice([0, 1], len(X_))
    result = obj.get_custom_quality_metrics(X_train=X_, y_test=y_, y_pred=y_, poisoning_idx=poisoning_idx,
                                            risk_values=risk_idx,
                                            with_risk=False)

    expected_idx = list(itertools.chain.from_iterable([base.CUSTOM_SCORE_OUTPUT_FULLNESS,
                                                       base.CUSTOM_SCORE_OUTPUT_POISONING_DEGREE,
                                                       base.CUSTOM_SCORE_OUTPUT_POISONING_STD,
                                                       base.CUSTOM_SCORE_OUTPUT_POISONING_RECALL,
                                                       base.CUSTOM_SCORE_OUTPUT_DIVERSITY]))

    # the default value
    assert len(result) == len(expected_idx)
    assert np.all(result.index == expected_idx)

    # it does not have risk-specific metrics.
    assert obj.get_custom_quality_metrics(X_train=X_, y_test=y_, y_pred=y_, poisoning_idx=poisoning_idx,
                                          risk_values=risk_idx,
                                          with_risk=True) is None
