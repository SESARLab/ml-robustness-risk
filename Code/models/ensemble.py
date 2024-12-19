import typing
import warnings

import joblib
import numbers
import numpy as np
from sklearn import ensemble,  preprocessing, clone, base as sk_base
from sklearn.utils import multiclass, validation

import pipe
import utils
from models import base

PIPELINE_LAST_NAME = 'last'


class EnsembleWithAssignmentPipeline(ensemble.VotingClassifier):

    def __init__(self, *,
                 base_estimator: sk_base.BaseEstimator,
                 data_point_assignment: pipe.ExtPipeline,
                 # N: int = 9,
                 n_jobs: typing.Optional[typing.Union[str, int]] = None,
                 flatten_transform: bool = True,
                 voting_type: base.VotingType = base.VotingType.HARD,
                 verbose: bool = False, **kwargs):

        # estimators = [ensemble.RandomForestClassifier() for _ in range(N)]
        self.base_estimator = base_estimator

        # first argument is estimators. But we cannot pass it, because we may know the
        # number of models only after clustering.
        super().__init__(None, n_jobs=n_jobs, flatten_transform=flatten_transform, verbose=verbose)
        # self.N = N
        self.n_jobs = n_jobs
        self.classes_ = None
        self.X_ = None
        self.y = None
        self.kwargs = kwargs
        # self.kwargs['N'] = N
        self.voting_type = voting_type
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.le_ = []
        self.estimators_ = None

        self.data_point_assignment = data_point_assignment
        # it contains the result of data point assignments.
        # This is useful for later inspection.
        self.assignment_ = None
        self.N_ = 0

    def _prepare_fit(self, X, y):
        validation.check_X_y(X, y)
        multiclass.check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                'Multilabel and multi-output classification is not supported.')
        validation.check_scalar(
            self.flatten_transform,
            name="flatten_transform",
            target_type=(numbers.Integral, np.bool_))

        self.le_ = preprocessing.LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)
        return transformed_y

    def _train_with_assignment(self, X, y, transformed_y):

        def fit_single(index, estimator_, X_, y_):
            to_use_X, to_use_y = X_, y_
            if len(to_use_X) == 0:
                warnings.warn(f'{self.data_point_assignment.name}: model {index} did not get any data point. Pick up the first.')
                to_use_X,to_use_y = X[0].reshape(1, -1), transformed_y[0].reshape(1, -1)
            return estimator_.fit(to_use_X, to_use_y)

        n_jobs = self.n_jobs
        if n_jobs == 'auto':
            n_jobs = self.N_

        try:

            # if assignments is 1d array then it's fine,
            # if it is a list then it means that we are doing a
            # multi-assignments, where data points get assigned to multiple partitions.
            with joblib.Parallel(n_jobs=n_jobs) as parallel:
                if isinstance(self.assignment_, np.ndarray):
                    assert len(self.assignment_) == len(X), f'{len(self.assignment_)} != {len(X)}, shape assignment: {self.assignment_.shape}'

                    # for i in range(self.N_):
                    #     if np.count_nonzero(self.assignment_ == i) == 0:
                    #         raise utils.EmptyModelException(f'Model {i} did not receive any data point. '
                    #                          f'Pipeline: {self.data_point_assignment.name}')


                    self.estimators_ = parallel(joblib.delayed(
                        fit_single)(
                        index=i,
                        estimator_=clone(self.base_estimator),
                        X_=X[np.where(self.assignment_ == i)],
                        y_=transformed_y[np.where(self.assignment_ == i)]
                    ) for i in range(self.N_))

                else:
                    # not to consider
                    assigned_X, assigned_y = utils.multi_assignment_to_X(X, y=y, assignment=self.assignment_, N=self.N_,
                                                                         n_jobs=n_jobs)
                    if isinstance(y, np.ndarray):
                        self.estimators_ = parallel(joblib.delayed(
                            fit_single)(
                            clone(self.base_estimator),
                            assigned_X[i],
                            assigned_y[i], )
                            # transformed_y[np.where(assignments == i)], )
                                                    for i in range(self.N_))
        except Exception as e:
            # not the best but at least we have some info
            print(f'Error during pipeline {self.data_point_assignment.name}')
            raise e

    def fit(self, X, y, sample_weight=None):
        transformed_y = self._prepare_fit(X, y)

        self.assignment_ = base.execute_pipeline(X, y, self.data_point_assignment)

        # now that we have run our assignments, we can extract N
        # from the last step of the pipeline.
        # last_step = self.data_point_assignment.named_steps[PIPELINE_LAST_NAME]
        # N = last_step.n_classes
        self.N_ = base.get_n_classes(self.data_point_assignment)

        self._train_with_assignment(X=X, y=y, transformed_y=transformed_y)

        return self

    @property
    def _weights_not_none(self):
        return self.weights

    def individual_predictions(self, X) -> np.ndarray:
        """
        Returns the predictions of each individual model on `X`.

        The output array is of shape `(len(X), len(N))`.
        :param X: the test set.
        :return: predictions of each model in the ensemble.
        """
        return self._predict(X)

    def hard_predictions_count(self, X) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array counting the number of models having predicted the correct class for each data point.

        The output is a 1d array whose length is the same of X. Each element represents the number of models
        having predicted the correct label.

        :return:
        - prediction
        - count (1d array)
        """
        # the shape of this object is (N_data_points, n_models)
        # this contains the predictions of each
        raw_predictions = self._predict(X)

        # retrieve weights from the last step.
        self.weights = base.get_weights(self.data_point_assignment, N=self.N_)
        # print(f'Using weights: {self.weights}')

        # this, instead, contains the *final* predictions of the ensemble.
        # taken directly from the code of sklearn.
        # (just to avoid doing a second pass).
        if self.voting_type == base.VotingType.HARD:
            maj = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                    axis=1,
                    arr=raw_predictions,)
        else:
            maj = np.argmax(self.predict_proba(X), axis=1)

        # the shape of maj is 1d.
        # (the following instruction does nothing at all).
        maj = self.le_.inverse_transform(maj)

        # now we expand these predictions to have the same shape of raw_predictions,
        # repeat repeats each individual element.
        # Let's assume that maj is [1, 0, 0, 0, 1]
        # it then becomes (assuming N=3)
        # [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]]
        # so the shape is (len(maj), N)
        expanded_predictions = np.repeat(maj, self.N_).reshape(raw_predictions.shape)

        # the shape of this object is (N_data_points)
        # np.count_nonzero(expanded_predictions == raw_predictions, axis=1) ->
        # it is pretty easy because we count for each row we check if it is equal to the expanded majority prediction
        # then we count the total.
        # So the result is: [number of models that match the majority prediction for each prediction].
        # We then divide by the number of models to have a normalized result.
        # i.e., 1 = complete match.
        count = np.count_nonzero(expanded_predictions == raw_predictions, axis=1) / self.N_
        return maj, count


if __name__ == '__main__':
    import assignments

    b = np.random.rand(100, 50)
    labels = np.array([0 for _ in range(50)] + [1 for _ in range(50)])

    # model = Ensemble(N=3, algo='md5', )
    model = EnsembleWithAssignmentPipeline(data_point_assignment=pipe.ExtPipeline(
        [pipe.Step('hash', assignments.Hasher(algo='md5')),
         pipe.Step('modulo', assignments.SingleValuedRouter(algo=assignments.SingleValuedRouterType.MODULO, N=7)),
         pipe.Step('last', assignments.AssignmentRoundRobinSmart(N=7))
         ]),
        base_estimator=ensemble.RandomForestClassifier(),
        n_jobs=5,
    )

    model.fit(b, labels)

    print(model.predict(np.array([b[0]])))
