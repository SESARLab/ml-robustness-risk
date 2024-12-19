import abc
import enum
import typing

import numpy as np
import pandas as pd
from sklearn import metrics

import utils

CUSTOM_SCORE_OUTPUT_FULLNESS = ['AVG_Fullness', 'STD_Fullness']
CUSTOM_SCORE_OUTPUT_POISONING_DEGREE = ['AvgPoisoningDegree']
CUSTOM_SCORE_OUTPUT_POISONING_STD = ['StdPoisoningSpread']
CUSTOM_SCORE_OUTPUT_POISONING_RECALL = ['PoisoningRecall']
CUSTOM_SCORE_OUTPUT_DIVERSITY = ['AvgDiversity']

CUSTOM_SCORE_ALL = CUSTOM_SCORE_OUTPUT_FULLNESS + CUSTOM_SCORE_OUTPUT_POISONING_DEGREE + \
                   CUSTOM_SCORE_OUTPUT_POISONING_STD + CUSTOM_SCORE_OUTPUT_POISONING_RECALL + \
                   CUSTOM_SCORE_OUTPUT_DIVERSITY


class OutputGuarantee(enum.Enum):
    ALWAYS = 'ALWAYS'
    NEVER = 'NEVER'
    DEPEND_ON_DATA = 'DEPEND_ON_DATA'


class AssignmentStrategy(enum.Enum):
    COMPRESS = 'COMPRESS'
    DISTRIBUTE = 'DISTRIBUTE'


class WeightingStrategy(enum.Enum):
    PROPORTIONAL = 'PROPORTIONAL'
    EXTREME = 'EXTREME'
    NONE = 'NONE'


QualityFunctionPair = typing.Tuple[typing.List[str],
typing.Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, typing.Optional[np.ndarray]], typing.Union[float, np.ndarray]]
]

class AbstractAssignment(utils.ContainerRngAndJobsMixin, abc.ABC):

    def __init__(self, N: int, n_jobs: typing.Optional[int] = None, copy: bool = False,
                 rng: typing.Optional[int] = None):
        super().__init__(n_jobs=n_jobs, rng=rng)
        self.copy = copy
        self.n_jobs = n_jobs
        self.n_clusters = N
        # for each partition, it counts how many points are assigned to it.
        self.assigned_to_each_ = np.zeros(N)
        # holds the true results of the assignment.
        self.assignment_ = None
        # self.len_X = -1
        self.used_weights = None

    def fill_assigned(self, result) -> np.ndarray:
        # assignment is of the form: [0, 1, 2, 3, 0, 1, 2, 3]
        # where each element *i* indicates the *i*-th data point is assigned to.
        self.assignment_ = result.copy()
        for i in range(self.n_clusters):
            self.assigned_to_each_[i] = np.count_nonzero(result == i)
        return self.assigned_to_each_

    def fullness_of_each(self) -> np.ndarray:
        return self.assigned_to_each_ / np.round(len(self.assignment_) / self.n_clusters)

    def fill_score(self, X, y_test, y_pred, poisoning_idx: typing.Optional[np.ndarray] = None,
                   risk_values: typing.Optional[np.ndarray] = None) -> typing.Tuple[float, float]:
        """
        Retrieves the *fill factor* of each model.

        The ideal number of points of a model is: round(len(X) / N).
        We thus retrieve, for each model, the number of assigned points divided
        by the ideal number.

        We then aggregate using avg and std.

        :return: avg and std of fill.

        In general, avg >= 1 indicates the most of the models are too full.
        """
        fraction_of_fill = self.fullness_of_each()
        return np.mean(fraction_of_fill), np.std(fraction_of_fill)

    # def fill_score(self) -> typing.Tuple[float, float]:
    #     """
    #     Retrieves the *fill factor* of each model.
    #
    #     The ideal number of points of a model is: round(len(X) / N).
    #     We thus retrieve, for each model, the number of assigned points divided
    #     by the ideal number.
    #
    #     We then aggregate using avg and std.
    #
    #     :return: avg and std of fill.
    #
    #     In general, avg >= 1 indicates the most of the models are too full.
    #     """
    #     fraction_of_fill = self.fullness_of_each()
    #     return np.mean(fraction_of_fill), np.std(fraction_of_fill)

    @property
    def n_classes(self) -> int:
        return self.n_clusters

    @staticmethod
    @abc.abstractmethod
    def guarantee_even_fill() -> OutputGuarantee:
        """
        True if it can guarantee that the created assignments will roughly have the same size.

        Some assignments cannot guarantee it, like those doing kmeans only. In fact, they can't
        guarantee that some clusters won't be empty.
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def strategy() -> AssignmentStrategy:
        pass

    def average_degree_of_poisoning(self, X, y_test, y_pred, poisoning_idx: np.ndarray,
                                    risk_values: typing.Optional[np.ndarray] = None):
        # it may happen that poisoning is all 0. It means we are trying on clean data, so
        # we ignore it.
        if np.count_nonzero(poisoning_idx) == 0:
            return np.nan
        count_all = np.bincount(self.assignment_)
        poisoned_assignments = self.assignment_[poisoning_idx.astype(bool)]
        count_poisoned = np.bincount(poisoned_assignments)
        # NOTE: it may happen that len(count_poisoned) < len(count_all)
        # if some models do not get any poisoned point. In such a case, we just pad.
        if len(count_poisoned) < len(count_all):
            count_poisoned = np.pad(count_poisoned, (0, len(count_all) - len(count_poisoned)))
        fraction = count_poisoned / count_all
        return np.mean(fraction)

    @abc.abstractmethod
    def poisoning_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray,
                        risk_values: typing.Optional[np.ndarray] = None) -> float:
        pass

    @abc.abstractmethod
    def poisoning_recall_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray, risk_values: typing.Optional[np.ndarray] = None) -> float:
        pass

    def number_of_points_to_each(self, X):
        return np.floor(len(X) / self.n_clusters)

    def diversity_score(self, X, y_test, y_pred, poisoning_idx: typing.Optional[np.ndarray] = None,
                        risk_values: typing.Optional[np.ndarray] = None) -> float:  # typing.Tuple[np.ndarray, np.ndarray]:
        # values = np.ones(len(self.assignment_))
        # silhouette_all = metrics.silhouette_samples(X, self.assignment_)
        # for i in range(self.n_clusters):
        #     values[i] = np.mean(silhouette_all[self.assignment_ == i])
        # return np.mean(values), np.std(values)
        return metrics.calinski_harabasz_score(X, self.assignment_)

    def get_weights(self):
        self.used_weights = self._get_weights()
        if np.count_nonzero(self.used_weights) == 0:
            self.used_weights = np.repeat(1 / self.n_clusters, self.n_clusters)
        return self.used_weights

    def _get_weights(self):
        return np.repeat(1 / self.n_clusters, self.n_clusters)

    @abc.abstractmethod
    def fit_transform(self, X, y=None):
        pass

    def get_custom_quality_metrics(self, X_train, y_pred, y_test, poisoning_idx: np.ndarray,
                                   risk_values: typing.Optional[np.ndarray] = None,
                                   with_risk: bool = False
                                   ) -> typing.Optional[pd.Series]:
        to_call = self.quality_metrics_not_risk if not with_risk else self.quality_metrics_risk
        funcs = to_call()
        if len(funcs) == 0:
            return None
        result = []
        for output_columns, func in funcs:
            # raw_output = func(y, poisoning_idx, risk_values)
            raw_output = func(X=X_train, y_pred=y_pred, y_test=y_test,
                              poisoning_idx=poisoning_idx, risk_values=risk_values)
            if not isinstance(raw_output, (np.ndarray, tuple, list)):
                raw_output = [raw_output]
            single_output = pd.Series(raw_output, index=output_columns)
            result.append(single_output)
        return pd.concat(result)

    def quality_metrics_not_risk(self) -> typing.List[QualityFunctionPair]:
        """
        Each child class declares which function it uses for quality and that's it. These are the default ones.
        :return:
        """
        return [
            (CUSTOM_SCORE_OUTPUT_FULLNESS, self.fill_score),
            (CUSTOM_SCORE_OUTPUT_POISONING_DEGREE, self.average_degree_of_poisoning),
            (CUSTOM_SCORE_OUTPUT_POISONING_STD, self.poisoning_score),
            (CUSTOM_SCORE_OUTPUT_POISONING_RECALL, self.poisoning_recall_score),
            (CUSTOM_SCORE_OUTPUT_DIVERSITY, self.diversity_score)
        ]

    def quality_metrics_risk(self) -> typing.List[QualityFunctionPair]:
        """
        Each child class declares which function it uses for quality and that's it. These are the default ones.
        :return:
        """
        return []


class AbstractAssignmentSpread(AbstractAssignment, abc.ABC):
    """
    Base class for assignments following a "spread" strategy.
    """

    def poisoning_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray,
                        risk_values: typing.Optional[np.ndarray] = None) -> float:
        """
        Defined as the standard deviation wrt to the *expected* (normalized) number
        of poisoned data points to be assigned to each model

        :param y_test:
        :param y_pred:
        :param y_test:
        :param X:
        :param risk_values:
        :param poisoning_idx:
        :return:
        """
        poisoning_idx_ = poisoning_idx.astype(bool)

        poisoned_count = np.count_nonzero(poisoning_idx_)
        # if there are no poisoned data points, we return the best value, 0 (i.e., no dispersion).
        if poisoned_count == 0:
            return 0.0
        # let's first retrieve the expected value. We always round down.
        # This is the ideal number of poisoned data points that should be assigned to each model.
        expected_number = np.floor(poisoned_count / self.n_clusters)
        # This is the *fraction* of poisoned data points that should be assigned to each model.
        expected_fraction = expected_number / poisoned_count
        # now, we count for each model how many poisoned data points are assigned to it.
        assignment_poisoned = self.assignment_[poisoning_idx_]
        # now let's count how many poisoned data points are assigned to each model.
        count = np.bincount(assignment_poisoned)
        # now, let's normalize this value wrt to the total number of poisoned data points.
        fraction = count / poisoned_count
        # and finally we retrieve the std (i.e., the squared root of the mean of the variance)
        std = np.sqrt(np.mean(np.abs(fraction - expected_fraction) ** 2))
        return std

    def poisoning_recall_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray, risk_values: typing.Optional[np.ndarray] = None) -> float:
        #
        # # if we do not have risk values, recall is nan.
        # if risk_values is None:
        #     return np.nan

        poisoning_idx = poisoning_idx.astype(bool)

        # let's make it simple: if we are working without any poisoned points, the recall is 1.
        if len(np.argwhere(poisoning_idx).flatten()) == 0:
            return 1.0

        # what we look for here is ensuring that poisoned points are (almost) evenly spread.
        # now, retrieve the assignments of poisoned points.
        assignment_of_poisoned = self.assignment_[poisoning_idx]
        ideal_spreading_up = np.ceil(len(assignment_of_poisoned) / self.n_clusters).astype(int)

        # now for each model we count how many poisoned data points it received.
        # in the 0-th position we have the number of poisoned points assigned to
        # the 0-th model, and so on.
        poisoned_assignment_of_each = np.bincount(assignment_of_poisoned)
        # now for each count we retrieve: min(count/ideal, 1)
        # this way if there are more points than the ideal the result for that model is 1 nevertheless,
        # and it will be penalized in other models
        div_up = poisoned_assignment_of_each / ideal_spreading_up
        # we don't need to use both lower and upper bound, in any case one model will be
        # penalized and one favored, so let's take the highest and that's it.
        # here, we apply the min using this trick.
        # for values <= 1.0, we keep what is there.
        div_min = np.where(div_up > 1.0, 1.0, div_up)
        # and finally we return the worst value, that is, min.
        return np.min(div_min)


class AbstractAssignmentSqueeze(AbstractAssignment, abc.ABC):

    def poisoning_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray,
                        risk_values: typing.Optional[np.ndarray] = None) -> float:
        """
        (normalized) number of poisoned data points not assigned to the required models
        :param X:
        :param y_test:
        :param y_pred:
        :param poisoning_idx:
        :param risk_values:
        :return:
        """

        poisoning_idx_ = poisoning_idx.astype(bool)

        poisoned_count = np.count_nonzero(poisoning_idx_)
        # if no poisoning, we have the perfect dispersion (i.e., the lack thereof).
        if poisoned_count == 0:
            return 0.0

        # ideal number of models to fairly assign data points.
        number_of_each_model = self.number_of_points_to_each(poisoning_idx_)
        # this gives us the number of models we are going to fill "completely".
        # It can be 0, in case even the first model is not going to be filled completely. In that case,
        # we set it to 1.
        n_models_to_fill_completely = max(int(np.floor_divide(poisoned_count, number_of_each_model)), 1)
        # now let's see where those poisoned points have been assigned to.
        assignment_of_poisoned = self.assignment_[poisoning_idx_]
        # and we extract a subset: *of the poisoned data points assigned to models outside the minimum
        # number of models*.
        assignment_of_poisoned_outside = assignment_of_poisoned[assignment_of_poisoned > n_models_to_fill_completely]
        # we don't do any bincount: all these points are wrong, so we just count how many of it are mis-assigned
        # and normalize this count.
        normalized_count_of_poisoned_outside = len(assignment_of_poisoned_outside) / poisoned_count
        return normalized_count_of_poisoned_outside

    def poisoning_recall_score(self, X, y_test, y_pred, poisoning_idx: np.ndarray,
                               risk_values: typing.Optional[np.ndarray] = None) -> float:

        # if risk is not available...
        if risk_values is None:
            return np.nan

        # let's make it simple: if we are working without any poisoned points, the recall is 1.
        if len(np.argwhere(poisoning_idx)) == 0:
            return 1.0

        poisoning_idx = poisoning_idx.astype(bool)
        # can it be reduced to the true recall? Na.
        # we retrieve the number of required models according to the poisoned points.
        # like "10 poisoned points and N=3, then I need to fill 4 models. 3 of them completely".
        ideals_n_models_to_be_filled = int(np.floor_divide(np.count_nonzero(poisoning_idx),
                                                           int(np.floor(
                                                               len(self.assigned_to_each_) / self.n_clusters))))

        # now, retrieve the assignments of poisoned points.
        assignment_of_poisoned = self.assignment_[poisoning_idx]
        # because we begin to fill from the lowest.
        if ideals_n_models_to_be_filled > 0:
            correct_assignment = np.count_nonzero(assignment_of_poisoned < ideals_n_models_to_be_filled)
        else:
            # in some cases we may get ideals_n_models_to_be_filled = 0, like 10 poisoned points and 20 models.
            # in this case we look for those assigned to the first model only.
            correct_assignment = np.count_nonzero(assignment_of_poisoned == 0)
        return correct_assignment / len(np.argwhere(poisoning_idx).flatten())


def trim_assignment(assignment, len_second: int):
    assignment = assignment[:-(len(assignment) - len_second)]
    return assignment
