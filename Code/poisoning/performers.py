import abc
import typing

import numpy as np

import utils
from . import base

T = typing.TypeVar('T')
V = typing.TypeVar('V')


TPoisoningInput = typing.TypeVar('TPoisoningInput', bound=base.AbstractPerformInfo)


class AbstractPerformer(abc.ABC, typing.Generic[TPoisoningInput]):

    def __init__(self, copy: bool = True):
        self.copy = copy
        self.idx_of_poisoned_data_points: np.ndarray = -np.ones(1)

    def fit(self,  X, y, specific_args: TPoisoningInput) -> "AbstractPerformer":
        pass

    @abc.abstractmethod
    def transform(self, X, y, selected_idx: typing.List[np.ndarray], specific_args: TPoisoningInput):
        pass

    @staticmethod
    @abc.abstractmethod
    def modified_parts() -> base.ModifiedPartOfPoints:
        pass


class AbstractPerformerLabelOnly(AbstractPerformer, abc.ABC):

    @staticmethod
    def modified_parts() -> base.ModifiedPartOfPoints:
        return base.ModifiedPartOfPoints.y


class PerformerLabelFlippingMonoDirectional(AbstractPerformerLabelOnly):

    def transform(self, X, y, selected_idx: typing.List[np.ndarray], specific_args: base.PerformInfoMonoDirectional):
        y_ = utils.copy_if(y, self.copy)

        # now we have to handle specific situations.
        # situation 1: we have 1 selected idx comprising *all* data points, not only the ones of the target class.
        skip_check = False
        if len(selected_idx) == 1 and len(selected_idx[0]) == len(X):
            # we need to extract only the relevant data points.
            _sub_idx = selected_idx[0]
            # we need to permute y as well, otherwise indexing fails.
            permuted_y = y[_sub_idx]
            _sub_idx = _sub_idx[permuted_y == specific_args.from_label]
        elif len(selected_idx) == 1 and len(selected_idx[0]) < len(X):
            # situation 2: it already contains the label to poison.
            _sub_idx = selected_idx[0]
            # skip_check = True
        elif len(selected_idx) == 2:
            # situation 3: idx are separated class by class, so we only take the
            # correct one.
            _sub_idx = selected_idx[specific_args.from_label]
            # skip_check = False
        else:
            raise ValueError(f'Unmanaged combination: len(selected_idx): {len(selected_idx)}, args: {specific_args}')

        n_data_points = specific_args.get_number_of_data_points(total_number=len(X))

        self.idx_of_poisoned_data_points = _sub_idx[:n_data_points]

        if not skip_check and not np.all(y_[self.idx_of_poisoned_data_points] == specific_args.from_label):
            raise ValueError(f'{self.__class__.__name__}: error in selecting data points, '
                             f'the selected ones are not of "from_label", '
                             f'correct ones: {y_[self.idx_of_poisoned_data_points] == specific_args.from_label}')
        y_[self.idx_of_poisoned_data_points] = specific_args.to_label

        if np.count_nonzero(y != y_) != n_data_points:
            raise ValueError(f'Safe check failed "np.count_nonzero(y != y_) != n_data_points": '
                             f'{np.count_nonzero(y != y_)} data points have changed, '
                             f'but {n_data_points} is expected. '
                             f'Len(x): {len(X)}, perc: {specific_args.perc_data_points}')

        return X, y_

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return True


class PerformerLabelFlippingBiDirectional(AbstractPerformerLabelOnly):

    def transform(self, X, y, selected_idx: typing.List[np.ndarray],
                  specific_args: base.PerformInfoBiDirectionalMirrored):
        y_ = utils.copy_if(y, self.copy)

        _sub_idx = {}

        if len(selected_idx) == 1:
            # all in a single array. Need to re-index y as well.
            permuted_y = y[selected_idx[0]].astype(int)
            # situation 1: indexes are mixed up, so we need to separate them.
            _sub_idx['a'] = selected_idx[0][permuted_y == specific_args.label_a]
            _sub_idx['b'] = selected_idx[0][permuted_y == specific_args.label_b]
        elif len(selected_idx) == 2:
            # already separated, but we are not sure whether the first one
            # list corresponds to label a or b. So we check.
            a_idx = 0
            b_idx = 1
            if int(y_[selected_idx[0][0]]) == specific_args.label_b:
                a_idx = 1
                b_idx = 0

            _sub_idx['a'] = selected_idx[a_idx]
            _sub_idx['b'] = selected_idx[b_idx]

        else:
            raise ValueError(f'Unmanaged combination: len(selected_idx): {len(selected_idx)}, args: {specific_args}')

        n_data_points = specific_args.get_number_of_data_points(total_number=len(X))

        n_data_points_a = int(np.ceil(n_data_points/2))
        n_data_points_b = int(np.floor(n_data_points/2))

        selected_idx_a = _sub_idx['a'][:n_data_points_a]
        selected_idx_b = _sub_idx['b'][:n_data_points_b]

        # now replace with label_a
        self.idx_of_poisoned_data_points = np.concatenate([selected_idx_a, selected_idx_b])

        y_[selected_idx_a] = specific_args.label_b
        y_[selected_idx_b] = specific_args.label_a

        assert np.count_nonzero(y != y_) == n_data_points_a + n_data_points_b

        return X, y_

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return True



