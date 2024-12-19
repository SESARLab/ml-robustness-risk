import typing

import numpy as np

import utils
from . import base
from .base import QualityFunctionPair


class AssignmentRoundRobinBlind(base.AbstractAssignmentSpread):
    """
    (Random) roundrobin assignments.

    It uses `np.tile` to perform assignments, and then, if requested, it shuffles the assignments.
    """

    #
    # @staticmethod
    # def strategy() -> base.AbstractAssignmentStrategy:
    #     return base.AssignmentStrategySpread()

    @staticmethod
    def strategy() -> base.AssignmentStrategy:
        return base.AssignmentStrategy.DISTRIBUTE

    def __init__(self, N: int, n_jobs: typing.Optional[int] = None, rng: typing.Optional[int] = None,
                 shuffle: bool = False):
        super().__init__(n_jobs=n_jobs, rng=rng, N=N)
        self.shuffle = shuffle

    def fit_transform(self, X, y=None):
        X_ = X

        assignment = np.tile(np.arange(self.n_clusters), int(np.ceil(len(X_) / self.n_clusters))).astype(int)
        if len(assignment) > len(X_):
            assignment = base.trim_assignment(assignment, len_second=len(X_))
        # shuffle if required
        if self.shuffle:
            # shuffling is done in place.
            self.rng.shuffle(assignment)

        self.fill_assigned(result=assignment)

        return assignment

    @property
    def n_classes(self) -> int:
        return self.n_clusters

    @staticmethod
    def guarantee_even_fill() -> base.OutputGuarantee:
        return base.OutputGuarantee.ALWAYS


class AssignmentRoundRobinSmart(base.AbstractAssignmentSpread):
    """
    This class works only in the case of 1D due to the fact that we are replacing each data point
    with its cluster center.

    A more sophisticated version will be added

    Note that it **must** receives as input values already quantized (e.g., cluster labels "high", "low", ...).
    """

    # @staticmethod
    # def strategy() -> base.AbstractAssignmentStrategy:
    #     return base.AssignmentStrategySpread()

    @staticmethod
    def strategy() -> base.AssignmentStrategy:
        return base.AssignmentStrategy.DISTRIBUTE

    def __init__(self, N: int, n_jobs: typing.Optional[int] = None, rng: typing.Optional[int] = None):
        super().__init__(n_jobs=n_jobs, rng=rng, N=N)
        self.aranged_cluster = np.arange(self.n_clusters)

    def fit_transform(self, X, y=None):
        X_ = utils.to_1d_or_raise(X)

        result = np.ones(len(X_)).astype(int)

        # since we are receiving in input values that are already quantized
        # e.g., [1, 3, 0, ...], we use those values to determine the number of partitions.
        # For instance in the above example: 3
        retrieved_n_values = np.unique(X_)

        # now we loop over the cluster centers starting from the riskier to less risky.
        retrieved_n_values = np.flip(np.sort(retrieved_n_values))

        for value_idx in retrieved_n_values:
            # find the number of data points to assign to each model of the ensemble, rounding down.
            involved_data_points = X_[X_ == value_idx]

            n_of_each_model = int(np.ceil(len(involved_data_points) / self.n_clusters))

            assignment = np.tile(self.aranged_cluster, n_of_each_model)

            # now, we basically have something like [0, 1, 2, 3, 0, 1, 3]
            # that we basically "pair" with the data points we are operating on.
            # Turns out that this sequence might not have the exact length of the data points to
            # assign, so we need to either cut it because we are using ceiling (round up).
            # Let me see. (cit. Tool).
            if len(assignment) > len(involved_data_points):
                # need to trim.
                assignment = base.trim_assignment(assignment=assignment,
                                                  len_second=len(involved_data_points))

            result[X_ == value_idx] = assignment

        self.fill_assigned(result=result)

        return result

    @staticmethod
    def guarantee_even_fill() -> base.OutputGuarantee:
        return base.OutputGuarantee.ALWAYS


CUSTOM_METRICS_NAME_DEGREE_OF_POISONING = ['DegPoisoning']
CUSTOM_METRICS_NAME_DEGREE_OF_RISK = ['DegRisk']
CUSTOM_METRICS_MODELS_SIZE = ['Size_First', 'Size_Other']


class AssignmentSqueezeSink(base.AbstractAssignmentSqueeze):
    """
    Squeeze-type assignment where all points with the highest risk value are assigned to
    the first model in any case. Round-robin then assigns the remaining points in a fair
    way to the remaining partitions.
    """

    @staticmethod
    def guarantee_even_fill() -> base.OutputGuarantee:
        return base.OutputGuarantee.DEPEND_ON_DATA

    # @staticmethod
    # def strategy() -> base.AbstractAssignmentStrategy:
    #     return base.AssignmentStrategyCompress()
    @staticmethod
    def strategy() -> base.AssignmentStrategy:
        return base.AssignmentStrategy.COMPRESS

    def degree_of(self, risk_idx):
        count_assigned_to_first = len(np.where(self.assignment_ == 0)[0])
        poisoned_assignments = self.assignment_[risk_idx.astype(bool)]
        count_poisoned_assigned_to_first = len(np.where(poisoned_assignments == 0)[0])
        fraction = count_poisoned_assigned_to_first / count_assigned_to_first
        return fraction

    def degree_of_poisoning_first(self, *, X, y_test, y_pred, poisoning_idx: np.ndarray,
                                  risk_values: typing.Optional[np.ndarray] = None):
        """
        how many *TRULY* poisoned data points goes to the first model?
        """
        return self.degree_of(risk_idx=poisoning_idx)

    def degree_of_risky_first(self, *, X, y_test, y_pred, poisoning_idx: np.ndarray,
                              risk_values: typing.Optional[np.ndarray] = None):
        """
        how many *TRULY* poisoned data points goes to the first model?
        """
        if risk_values is None:
            raise ValueError('risk_values cannot be None')
        return self.degree_of(risk_idx=risk_values)

    def quality_metrics_risk(self) -> typing.List[QualityFunctionPair]:
        return super().quality_metrics_risk() + [
            (CUSTOM_METRICS_NAME_DEGREE_OF_POISONING, self.degree_of_poisoning_first),
            (CUSTOM_METRICS_NAME_DEGREE_OF_RISK, self.degree_of_risky_first)
        ]

    def quality_metrics_not_risk(self) -> typing.List[QualityFunctionPair]:
        return super().quality_metrics_risk() + [
            (CUSTOM_METRICS_MODELS_SIZE, self.score_models_size)
        ]

    def __init__(self, N: int,
                 weighting_strategy: base.WeightingStrategy = base.WeightingStrategy.NONE,
                 final_pass_algorithm_clazz: typing.Type[utils.Transformer] = None,
                 final_pass_kwargs: typing.Optional[dict] = None,
                 percentage_of_risky: typing.Optional[float] = None,
                 ):
        super().__init__(N)
        self.points_to_first = 0
        self.final_pass_clazz = final_pass_algorithm_clazz or AssignmentRoundRobinSmart
        self.final_pass_kwargs = final_pass_kwargs or {}
        # by default, we use 1 if not specified.
        self.percentage_of_risky = percentage_of_risky or 1.0
        self.inner_rr_: utils.Transformer = None
        self.weighting_strategy = weighting_strategy

    def _get_weights(self):
        if self.weighting_strategy == base.WeightingStrategy.PROPORTIONAL or \
                self.weighting_strategy == base.WeightingStrategy.EXTREME:
            return np.concatenate((np.zeros((1,)), np.repeat(1 / (self.n_clusters - 1), self.n_clusters - 1)))
        else:
            # use default strategy (equal).
            return super()._get_weights()

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X_ = X
        if not self.final_pass_clazz.guarantee_even_fill():
            raise ValueError(f'{self.__class__.__name__} requires a finalizer that can guarantee even fill.')
        if len(X_.shape) == 2 and X_.shape[1] == 1:
            # can flatten.
            X_ = X_.flatten()
        elif len(X_.shape) != 1:
            raise ValueError(f'{self.__class__.__name__} requires a 1d-shaped risk, got: {X_.shape}')

        if 1 < self.percentage_of_risky < 0:
            raise ValueError(f'{self.__class__.__name__} requires a percentage of risky points between 0 and 1')

        risk_idx = X_

        # we just create it.
        result = np.zeros(len(risk_idx)).astype(int)

        # issue: if we are training without poisoning, *everything* is assigned to the first model.
        if np.count_nonzero(risk_idx) != 0:

            # everything with a risky value is assigned to first model.
            # how many different values for the risk do we have?
            n_unique_values = np.unique(risk_idx)
            # we sort the risk values from the highest to the worst.
            n_unique_values = np.flip(np.sort(n_unique_values))

            # the next one is a boolean array where True at the i-th position
            # indicates that the i-th element has the highest risk value.
            # here we will have: [True, True, True, False, False, ...]
            # worst_points_idx: np.ndarray = risk_idx == n_unique_values[0]
            worst_points_idx = np.where(risk_idx == n_unique_values[0])[0]

            # now, we select a subset of these points according to the given percentage.
            # 1. shuffle the indices
            # 2. extract the first N.
            first_n = int(np.round(len(worst_points_idx) * self.percentage_of_risky))
            # in place-shuffling.
            self.rng.shuffle(worst_points_idx)
            worst_points_idx = worst_points_idx[:first_n]

            # and that's it, here we assign all worst points to the first model.
            result[worst_points_idx] = 0
            # set the counter
            # self.points_to_first = np.count_nonzero(worst_points_idx)
            self.points_to_first = first_n
            # TODO may be helpful for the future
            # print(f'{self.percentage_of_risky}: {self.points_to_first} -> {self.points_to_first/len(X)}')

            # now we complete the final pass where we fill the remaining models.
            # free_idx = np.logical_not(worst_points_idx)
            free_idx = np.where(risk_idx != n_unique_values[0])[0]

            # all but the first model are available.
            self.final_pass_kwargs['N'] = self.n_clusters - 1
            incrementer = 1
        else:
            # all points are available
            free_idx = np.arange(len(X_))
            self.final_pass_kwargs['N'] = self.n_clusters
            incrementer = 0

        available_points = X_[free_idx]
        self.inner_rr_ = self.final_pass_clazz(**self.final_pass_kwargs)
        remaining_assignment = self.inner_rr_.fit_transform(X=available_points)

        # the result is incremented by 1 because the first model cannot be used
        # (only if we are in the non-clean scenario. In that case, we increment by 0).
        remaining_assignment += incrementer

        # now we need to fuse remaining_assignment to result.
        result[free_idx] = remaining_assignment

        self.assigned_to_each_ = self.fill_assigned(result=result)

        return result

    def score_models_size(self, *, X, y_test, y_pred, poisoning_idx: np.ndarray,
                          risk_values: typing.Optional[np.ndarray] = None) -> typing.Tuple[float, float]:
        """
        Return the (normalized) size of the first and second models.
        Note that the size of the second model acts as a proxy for the size of all other models.
        :param X:
        :param y_test:
        :param y_pred:
        :param poisoning_idx:
        :param risk_values:
        :return:
        """
        return (self.points_to_first / len(self.assignment_),
                np.count_nonzero(self.assignment_ == 1) / len(self.assignment_))
