# import numpy as np
# import pandas as pd
#
# import const
# from . import base, experiment_ensemble_plain, experiment_ensemble_risk
#
#
# def standard_model_quality_agg(info: base.ExpInfo) -> pd.Series:
#     return info.prepend_to(pd.Series(
#         np.concatenate([np.ones(len(base.METRICS_NAME)),
#                         np.zeros(len(base.METRICS_NAME))]),
#         index=[f'{const.PREFIX_AVG}({col})' for col in
#                base.METRICS_NAME] + [
#                   f'{const.PREFIX_STD}({col})' for col in
#                   base.METRICS_NAME]))


# def _test_pair_result(
#         pair: typing.Union[
#             experiment_ensemble_plain.CleanPoisonedOutputPair, experiment_ensemble_risk.CleanPoisonedOutputPair],
#         info_poisoned, info_clean):
#     got_model_quality, got_delta = pair.delta()
#
#     # y is 3 (pipeline, perc_points, perc_features) + 2 * (metrics). The reason of the multiplication
#     # is: we have both avg and std
#     assert got_model_quality.shape == (len(info_poisoned) + 1, 3 + len(base.METRICS_NAME) * 2)
#     # same as above, but here we don't have std.
#     assert got_delta.shape == (len(info_poisoned), 3 + len(base.METRICS_NAME))
#
#     # these checks are very picky. but whatever...
#     assert np.all(got_delta[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#     assert np.all(got_delta[base.KEY_PERC_DATA_POINTS] == [info.perc_points for info in info_poisoned])
#     assert np.all(got_delta[base.KEY_PERC_FEATURES] == [info.perc_features for info in info_poisoned])
#
#     assert np.all(got_model_quality[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#     assert np.all(got_model_quality[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_points] +
#                   [info.perc_points for info in info_poisoned])
#     assert np.all(got_model_quality[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_features] +
#                   [info.perc_features for info in info_poisoned])
#
#
# def _test_pair_result(
#         pair: typing.Union[
#             experiment_ensemble_plain.CleanPoisonedOutputPair, experiment_ensemble_risk.CleanPoisonedOutputPair],
#         info_poisoned, info_clean):
#     got_model_quality, got_delta = pair.delta()
#
#     # y is 3 (pipeline, perc_points, perc_features) + 2 * (metrics). The reason of the multiplication
#     # is: we have both avg and std
#     assert got_model_quality.shape == (len(info_poisoned) + 1, 3 + len(base.METRICS_NAME) * 2)
#     # same as above, but here we don't have std.
#     assert got_delta.shape == (len(info_poisoned), 3 + len(base.METRICS_NAME))
#
#     # these checks are very picky. but whatever...
#     assert np.all(got_delta[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#     assert np.all(got_delta[base.KEY_PERC_DATA_POINTS] == [info.perc_points for info in info_poisoned])
#     assert np.all(got_delta[base.KEY_PERC_FEATURES] == [info.perc_features for info in info_poisoned])
#
#     assert np.all(got_model_quality[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#     assert np.all(got_model_quality[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_points] +
#                   [info.perc_points for info in info_poisoned])
#     assert np.all(got_model_quality[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_features] +
#                   [info.perc_features for info in info_poisoned])


# def _test_pair_result_advanced(
#         pair: typing.Union[
#             experiments.experiment_common.AnalyzedOutputPairBase, experiments.experiment_common.AnalyzedOutputPairEnsemble],
#         info_clean: base.ExpInfo,
#         info_poisoned: typing.List[base.ExpInfo]
#     ):
#     """
#     we receive as input the info because they are the only thing that are important here.
#     :param pair:
#     :param info_clean:
#     :param info_poisoned:
#     :return:
#     """
#     # y is INFO_KEYS_SET (pipeline, perc_points, perc_features) + 2 * (metrics).
#     # The reason of the multiplication is: we have both avg and std for each metric.
#     assert pair.model_quality.shape == (len(info_poisoned) + 1, len(base.INFO_KEYS_SET) + len(base.METRICS) * 2)
#     # same as above, but here we don't have std.
#     assert pair.delta_self.shape == (len(info_poisoned), len(base.INFO_KEYS_SET) + len(base.METRICS))
#
#     # very picky checks...
#     assert np.all(pair.delta_self[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#     assert np.all(pair.delta_self[base.KEY_PERC_DATA_POINTS] == [info.perc_points for info in info_poisoned])
#     assert np.all(pair.delta_self[base.KEY_PERC_FEATURES] == [info.perc_features for info in info_poisoned])
#
#     source = [pair.model_quality]
#
#     # now, if we are receiving results referred to the ensemble,
#     # we also check delta_base.
#     if isinstance(pair, experiments.experiment_common.AnalyzedOutputPairEnsemble):
#         # we have the difference with respect to the base model
#         # across all percentages of poisoning (len(result_poisoned)) and clean as well (+1) -> first dimension.
#         # In the second dimension, we have INFO_KEYS_SET + the metrics on which we retrieve the difference.
#         assert pair.delta_base.shape == (len(info_poisoned) + 1, len(base.INFO_KEYS_SET) + len(base.METRICS))
#
#         source.append(pair.delta_base)
#
#     # now we check also for model quality and, if present, for delta_base.
#     # Note: these results are available also for the clean dataset, so we have one row more
#     # compared to delta_self.
#     for single_source in source:
#
#         assert np.all(single_source[base.KEY_PIPELINE_NAME] == info_poisoned[0].pipeline_name)
#         assert np.all(single_source[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_points] +
#                   [info.perc_points for info in info_poisoned])
#         assert np.all(single_source[base.KEY_PERC_DATA_POINTS] == [info_clean.perc_features] +
#                   [info.perc_features for info in info_poisoned])
