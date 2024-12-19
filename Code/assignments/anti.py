import abc
import math
import typing

import numpy as np
from scipy import spatial
from scipy.spatial import distance
from sklearn import preprocessing

import utils
from assignments import base


class AntiClusteringReference(base.AbstractAssignment, abc.ABC):

    def __init__(self, n_clusters: int, n_jobs: typing.Optional[int] = None, rng: typing.Optional[int] = None,
                 ):
        super().__init__(rng=rng, n_jobs=n_jobs)
        self.n_clusters = n_clusters
        self.categorical_columns: typing.List[int] = []
        self.numeric_columns: typing.List[int] = []
        self.distance_matrix_: np.ndarray = None

    def fit(self, X, y=None, **fit_params):
        # X_ = self._pre(X)
        # self.distance_matrix_ = self._distance_matrix(X)
        return self

    def transform(self, X, y):
        if len(self.numeric_columns) == 0 and len(self.categorical_columns) == 0:
            # they are all numeric.
            self.numeric_columns = np.arange(X.shape[1])
        self.distance_matrix_ = self._distance_matrix(X)

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X_ = X.copy()
        return preprocessing.MinMaxScaler().fit_transform(X_[:, self.numeric_columns])

    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        # todo this is slow for some reasons
        d = 0
        if len(self.categorical_columns) > 0:
            d = distance.squareform(distance.pdist(
                preprocessing.LabelEncoder().fit_transform(X[:, self.categorical_columns]), metric='hamming'))
        c = 0
        if len(self.numeric_columns) > 0:
            c = spatial.distance_matrix(X[:, self.numeric_columns], X[:, self.numeric_columns])
        return c + d

    @staticmethod
    def _post_process(X, cluster_assignment: np.ndarray) -> np.ndarray:
        components = utils.UnionFind(len(X))
        for j in range(len(X)):
            for i in range(0, j):
                if cluster_assignment[i][j] == 1:
                    components.union(i, j)
        out = [components.find(i) for i in range(len(X))]
        # the output does not necessary includes labels [0, ... n-1],
        # as some labels may be skipped, e.g., [0, 1, 2, 4].
        # We use LabelEncoder to fix the issue.
        return preprocessing.LabelEncoder().fit_transform(out)

    @property
    def n_classes(self) -> int:
        return self.n_clusters


class AntiClusteringReferenceSwapHeuristic(AntiClusteringReference, abc.ABC):

    def __init__(self, n_clusters: int, n_jobs: typing.Optional[int] = None, rng: typing.Optional[int] = None):
        super().__init__(rng=rng, n_jobs=n_jobs, n_clusters=n_clusters)

    @staticmethod
    def _get_exchanges(cluster_assignment: np.ndarray, i: int) -> np.ndarray:
        return np.nonzero(np.invert(cluster_assignment[i]))[0]

    @staticmethod
    def _swap(cluster_assignment: np.ndarray, i: int, j: int) -> np.ndarray:
        cluster_assignment = cluster_assignment.copy()
        tmp1 = cluster_assignment[i,].copy()
        tmp2 = cluster_assignment[:, i].copy()
        cluster_assignment[i,] = cluster_assignment[j,]
        cluster_assignment[:, i] = cluster_assignment[:, j]
        cluster_assignment[j,] = tmp1
        cluster_assignment[:, j] = tmp2
        cluster_assignment[i, j] = False
        cluster_assignment[j, i] = False
        return cluster_assignment

    def _get_random_clusters(self, num_elements: int) -> np.ndarray:
        # Using UnionFind to generate random anti-clusters. The first num_groups elements are guaranteed
        # to be roots of each their own component. All other elements are assigned a random root.
        initial_clusters = [i % self.n_clusters for i in range(num_elements - self.n_clusters)]
        self.rng.shuffle(initial_clusters)
        initial_clusters = list(range(self.n_clusters)) + initial_clusters
        uf_init = utils.UnionFind(len(initial_clusters))
        for i, cluster_ in enumerate(initial_clusters):
            uf_init.union(i, cluster_)
        cluster_assignment = np.array(
            [[uf_init.connected(i, j) for i in range(num_elements)] for j in range(num_elements)]
        )
        return cluster_assignment


class AntiClusteringReferenceSA(AntiClusteringReferenceSwapHeuristic):

    def __init__(self, n_clusters: int, n_jobs: typing.Optional[int] = None, rng: typing.Optional[int] = None,
                 alpha: float = 0.9, iterations: int = 3000, starting_temperature: float = 10
                 ):
        super().__init__(rng=rng, n_jobs=n_jobs, n_clusters=n_clusters)
        self.alpha = alpha if alpha is not None else 0.9
        self.iterations = iterations if iterations is not None else 2000
        self.starting_temperature = starting_temperature if starting_temperature is not None else 10

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, iterations={self.iterations}, ' \
               f'starting_t={self.starting_temperature})'

    def transform(self, X, y=None):
        super().transform(X, y)
        cluster_assignment = self._get_random_clusters(num_elements=len(self.distance_matrix_))
        temperature = self.starting_temperature
        objective = self._calculate_objective(cluster_assignment, self.distance_matrix_)
        for iteration in range(self.iterations):
            # Select random element
            # i = self.rng.randint(0, len(self.distance_matrix_) - 1)
            i = self.rng.integers(0, len(self.distance_matrix_) - 1)
            # Get possible swaps
            possible_exchanges = self._get_exchanges(cluster_assignment, i)
            if len(possible_exchanges) == 0:
                continue
            # Select random possible swap.
            # j = possible_exchanges[self.rng.randint(0, len(possible_exchanges)-1)]
            j = possible_exchanges[self.rng.integers(0, len(possible_exchanges) - 1)]
            new_cluster_assignment = self._swap(cluster_assignment, i, j)
            new_objective = self._calculate_objective(new_cluster_assignment, self.distance_matrix_)
            # Select solution as current if accepted
            if self._accept(new_objective - objective, temperature):
                objective = new_objective
                cluster_assignment = new_cluster_assignment
            # Cool down temperature
            temperature = temperature * self.alpha
        return self._post_process(X, cluster_assignment)

    @staticmethod
    def _calculate_objective(cluster_assignment: np.ndarray, distance_matrix: np.ndarray) -> float:
        """
        Calculate objective value
        :param cluster_assignment: Cluster assignments
        :param distance_matrix: Distance matrix
        :return: Objective value
        """
        return np.multiply(cluster_assignment, distance_matrix).sum()

    def _accept(self, delta: float, temperature: float) -> bool:
        """
        Simulated annealing acceptance function. Notice d/t is negated because this is a maximisation problem.
        :param delta: Difference in objective
        :param temperature: Current temperature
        :return: Whether the solution is accepted or not.
        """
        return delta >= 0 or math.exp(delta / temperature) >= self.rng.uniform(0, 1)
#