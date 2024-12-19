import warnings

import numpy as np
import pytest
import sklearn
from sklearn import datasets, metrics, model_selection, tree

from . import monolithic_oracle, ensemble
import assignments
import pipe


@pytest.mark.parametrize('wrapped', [
    tree.DecisionTreeClassifier(),
    ensemble.EnsembleWithAssignmentPipeline(data_point_assignment=pipe.ExtPipeline(name='pipe', steps=[
        pipe.Step('hash', assignments.Hasher(algo='md5')),
        pipe.Step('router', assignments.SingleValuedRouter(algo=assignments.SingleValuedRouterType.MODULO, N=5)),
        pipe.Step('last', assignments.AssignmentRoundRobinSmart(N=5))]
    ), base_estimator=tree.DecisionTreeClassifier()),
])
def test_training(wrapped):
    # X, y =
    X_train, X_test, y_train, y_test = model_selection.train_test_split(*datasets.make_classification(n_samples=100))

    poisoning_info = np.random.default_rng().choice(1, size=len(X_train))

    oracle_poisoned = monolithic_oracle.EstimatorWithOracle(sklearn.clone(wrapped), poisoning_info=poisoning_info)
    oracle_poisoned.fit(X_train, y_train)
    oracle_non_poisoned = monolithic_oracle.EstimatorWithOracle(sklearn.clone(wrapped))
    oracle_non_poisoned.fit(X_train, y_train)

    y_pred_poisoned = oracle_poisoned.predict(X_test)
    y_pred_non_poisoned = oracle_non_poisoned.predict(X_test)

    # # check that the accuracy is different.
    assert metrics.accuracy_score(y_test, y_pred_poisoned) != metrics.accuracy_score(y_test, y_pred_non_poisoned) or \
           metrics.recall_score(y_test, y_pred_poisoned) != metrics.recall_score(y_test, y_pred_non_poisoned) or \
           metrics.precision_score(y_test, y_pred_poisoned) != metrics.precision_score(y_test, y_pred_non_poisoned)
    assert np.all(oracle_poisoned.poisoning_info == poisoning_info)
    assert oracle_non_poisoned.poisoning_info is None


@pytest.mark.parametrize('wrapped', [
    tree.DecisionTreeClassifier(),
    ensemble.EnsembleWithAssignmentPipeline(data_point_assignment=pipe.ExtPipeline(name='pipe', steps=[
        pipe.Step('hash', assignments.Hasher(algo='md5')),
        pipe.Step('router', assignments.SingleValuedRouter(algo=assignments.SingleValuedRouterType.MODULO, N=3)),
        pipe.Step('last', assignments.AssignmentRoundRobinSmart(N=3))]
    ), base_estimator=tree.DecisionTreeClassifier()),
])
def test_training_warning(wrapped):
    X, y = datasets.make_classification()
    oracle_non_poisoned = monolithic_oracle.EstimatorWithOracle(wrapped=wrapped)
    # check that the warning (raised when there is no poisoning_info) is actually raised
    with warnings.catch_warnings(record=True) as w:
        oracle_non_poisoned.fit(X, y)
        assert len(w) == 1
        assert 'invoked without poisoning_info' in str(w[-1].message)
