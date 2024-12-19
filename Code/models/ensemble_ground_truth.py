import typing

import numpy as np

from . import base, ensemble
import pipe


class EnsembleWithAssignmentPipelineGroundTruth(ensemble.EnsembleWithAssignmentPipeline):
    """
    Extension of the normal ensemble with pipeline where the assignment is retrieved
    using *ground truth risk*, i.e., according to poisoning information.
    """

    def __init__(self, *, base_estimator, data_point_assignment: pipe.ExtPipeline,
                 risk_ground_truth: typing.Optional[np.ndarray] = None,
                 n_jobs: typing.Optional[typing.Union[str, int]] = None,
                 flatten_transform: bool = True, verbose: bool = False, voting: str = 'hard', **kwargs):
        super().__init__(base_estimator=base_estimator, data_point_assignment=data_point_assignment, n_jobs=n_jobs,
                         flatten_transform=flatten_transform, verbose=verbose, voting=voting, **kwargs)
        self.risk_ground_truth = risk_ground_truth

    def fit(self, X, y, sample_weight=None):
        if self.risk_ground_truth is None:
            raise ValueError('Risk ground truth not set')

        transformed_y = self._prepare_fit(X, y)
        # use the pipeline with the risk ground truth.
        self.assignment_ = base.execute_pipeline(X=self.risk_ground_truth, y=y, p=self.data_point_assignment)
        self.N_ = base.get_n_classes(self.data_point_assignment)
        self._train_with_assignment(X=X, y=y, transformed_y=transformed_y)
        return self
