from sklearn import datasets, tree

import assignments
from . import ensemble
import pipe


def test_basic():
    X, y = datasets.make_classification()

    clf = ensemble.EnsembleWithAssignmentPipeline(data_point_assignment=pipe.ExtPipeline(
        [pipe.Step('hash', assignments.Hasher(algo='md5')),
         pipe.Step('last', assignments.SingleValuedRouter(algo='fibonacci', N=16))]),
        base_estimator=tree.DecisionTreeClassifier(),
    )

    clf.fit(X, y).predict(X)


def test_pred_count():
    pass