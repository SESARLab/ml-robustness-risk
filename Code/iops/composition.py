import typing

import numpy as np
from sklearn import preprocessing

import utils
from . import base, composition_pre_post


class IoPComposer:

    def __init__(self,
                 iop_clazz: typing.Type[base.IoP],
                 iop_kwargs: typing.Optional[typing.Dict] = None,
                 grouper_clazz: typing.Optional[typing.Type[composition_pre_post.AbstractGrouper]] = None,
                 grouper_kwargs: typing.Optional[typing.Dict] = None,
                 aggregator_clazz: typing.Optional[typing.Type[composition_pre_post.AbstractAggregator]] = None,
                 aggregator_kwargs: typing.Optional[typing.Dict] = None,
                 scaler_clazz: typing.Optional[typing.Type[utils.Transformer]] = None,
                 scaler_kwargs: typing.Optional[typing.Dict] = None,
                 ):
        self.iop_kwargs = iop_kwargs or {}
        self.grouper_kwargs = grouper_kwargs or {}
        self.finalizer_kwargs = aggregator_kwargs or {}

        self.iop_clazz = iop_clazz
        self.grouper_clazz = grouper_clazz or composition_pre_post.GrouperNoGroup
        self.aggregator_clazz = aggregator_clazz or composition_pre_post.AggregatorNoAggregator

        self.grouper: composition_pre_post.AbstractGrouper = None
        # self.iop = None
        # self.finalizer = None
        # self.scaler = None

        self.scaler_clazz = scaler_clazz or preprocessing.MinMaxScaler
        self.scaler_kwargs = scaler_kwargs or {}

    def fit_transform(self, X, y, **kwargs):
        self.grouper = self.grouper_clazz(**self.grouper_kwargs)

        X_scaled = X

        if self.grouper.requires_scaled_input:
            scaler = self.scaler_clazz(**self.scaler_kwargs)
            X_scaled = scaler.fit_transform(X_scaled)

        # first, apply the grouper.
        groups_idx = self.grouper.fit_transform(X_scaled, y, **kwargs)
        group_labels = np.unique(groups_idx)

        # output = np.zeros(X.shape[0])
        output_raw: typing.List[typing.Tuple[np.ndarray, np.ndarray]] = []
        output_shape = (0,)

        for group_id in np.nditer(group_labels):

            # apply the IoP on each individual group.
            iop = self.iop_clazz(**self.iop_kwargs)
            # do we need to normalize?
            if iop.step_.requires_scaled_input:
                scaler = self.scaler_clazz(**self.scaler_kwargs)
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            output_shape = iop.step_.output_shape

            result = iop.fit_transform(X_scaled[groups_idx == group_id], y[groups_idx == group_id])

            # # reshape result if needed.
            # if len(result.shape) == 1:
            #     result = result.reshape(-1, 1)

            # apply the finalizer on each individual group.
            finalizer = self.aggregator_clazz(**self.finalizer_kwargs)

            result = finalizer.fit_transform(X=result, y=y)
            if len(result.shape) == 1:
                result = result.reshape(-1, 1)

            # output[groups_idx == group_id] = result
            output_raw.append((groups_idx == group_id, result))

        # now prepare the final output
        output = np.zeros((len(X), *output_shape))
        for (idx, result_) in output_raw:
            output[idx] = result_

        return output
