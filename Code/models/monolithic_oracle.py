import typing
import warnings

import numpy as np
from sklearn import base as sk_base

import utils

TEstimator = typing.TypeVar('TEstimator', bound=utils.EstimatorProtocol)


class EstimatorWithOracle(sk_base.BaseEstimator):
    """
    Class wrapping an already initialized sklearn estimator training *on non-poisoned data points* only.
    When poisoning_info is None, it fall backs to the wrapped estimator (printing a warning).
    """

    def __init__(self, wrapped, poisoning_info: typing.Optional[np.ndarray] = None):
        self.wrapped_ = wrapped
        self.poisoning_info = poisoning_info

    def fit(self, X, y, poisoning_info: typing.Optional[np.ndarray] = None, **kwargs):
        if poisoning_info is not None and self.poisoning_info is None:
            self.poisoning_info = poisoning_info
        if self.poisoning_info is None:
            warnings.warn(f'{self.__class__.__name__} invoked without poisoning_info.'
                          f'Wrapping: {self.wrapped_}')
            return self.wrapped_.fit(X, y, **kwargs)
        else:
            return self.wrapped_.fit(X[~self.poisoning_info.astype(bool)], y[~self.poisoning_info.astype(bool)], **kwargs)

    def predict(self, X):
        return self.wrapped_.predict(X)

    def __repr__(self):
        return self.wrapped_.__repr__()
