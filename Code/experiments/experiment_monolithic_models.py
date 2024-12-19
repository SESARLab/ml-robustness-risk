import copy
import dataclasses
import os
import typing

import joblib
import numpy as np
import pandas as pd
import xarray as xr

import const
import models
from . import base, dataset_generator, experiment_common as common
import utils, utils_exp_post
from .experiment_common import TestSetType

TEstimator = typing.TypeVar('TEstimator', bound=utils.EstimatorProtocol)

class ExportConfigBaseModels(base.AbstractExportConfigWithDirectory):
    pass


@dataclasses.dataclass
class AnalyzedResultMonolithicModels:
    """
    Attributes:
          model_quality_oracle: typing.Dict[common.TestSetType, pd.DataFrame]

        model_quality_vanilla: typing.Dict[common.TestSetType, pd.DataFrame]

        delta_self_oracle: typing.Dict[common.TestSetType, pd.DataFrame]

        delta_self_vanilla: typing.Dict[common.TestSetType, pd.DataFrame]

        delta_ref_vanilla_oracle: typing.Dict[common.TestSetType, pd.DataFrame]
    """
    model_quality_oracle: typing.Dict[common.TestSetType, pd.DataFrame]
    model_quality_vanilla: typing.Dict[common.TestSetType, pd.DataFrame]
    delta_self_oracle: typing.Dict[common.TestSetType, pd.DataFrame]
    delta_self_vanilla: typing.Dict[common.TestSetType, pd.DataFrame]
    delta_ref_vanilla_oracle: typing.Dict[common.TestSetType, pd.DataFrame]

    @staticmethod
    def from_results(
            results_vanilla: typing.Sequence[common.CleanPoisonedOutputPair[common.TrainSingleOutputMonolithicWithRep]],
            results_oracled: typing.Sequence[common.CleanPoisonedOutputPair[common.TrainSingleOutputMonolithicWithRep]]
    ):
        model_names = []
        model_qualities_oracled = {t: [] for t in common.TestSetType}
        model_qualities_vanilla = {t: [] for t in common.TestSetType}
        delta_ref = {t: [] for t in common.TestSetType}
        delta_self_vanilla = {t: [] for t in common.TestSetType}
        delta_self_oracled = {t: [] for t in common.TestSetType}

        model_quality_oracled_df_s, model_quality_vanilla_df_s, delta_ref_vanilla_oracled_df_s = {}, {}, {}
        delta_self_vanilla_df_s, delta_self_oracle_df_s = {}, {}

        # for each model we have a list of results.
        # so, first, we aggregate the results of each model into a pd.DataFrame
        # In other words, we are iterating over the different models.
        for results_of_model_vanilla, results_of_model_oracled in zip(results_vanilla, results_oracled):

            model_names.append(results_of_model_vanilla.pipeline_name)

            for test_set_type in common.TestSetType:

                quality_vanilla_poisoned = []
                quality_oracled_poisoned = []

                for i, single_result in enumerate(results_of_model_vanilla.poisoned):
                    # for test_set_type, model_quality
                    quality_vanilla_poisoned.append(single_result.model_quality[test_set_type])

                for i, single_result in enumerate(results_of_model_oracled.poisoned):
                    quality_oracled_poisoned.append(single_result.model_quality[test_set_type])

                quality_vanilla_poisoned_df = pd.DataFrame(quality_vanilla_poisoned)
                quality_oracle_poisoned_df = pd.DataFrame(quality_oracled_poisoned)

                # now, we retrieve the delta.
                columns_to_consider_in_delta_ = common.columns_to_consider_in_delta(quality_vanilla_poisoned_df)

                VANILLA, ORACLE = 'vanilla', 'oracle'
                delta_self_results = {}

                # we retrieve delta self
                for k, src_poisoned, src_clean in [(VANILLA, quality_vanilla_poisoned_df, results_of_model_vanilla),
                                                   (ORACLE, quality_oracle_poisoned_df, results_of_model_oracled)]:
                    delta_self = src_poisoned[columns_to_consider_in_delta_] - \
                                 src_clean.clean.model_quality[test_set_type][columns_to_consider_in_delta_]
                    delta_self = base.add_info_to_df(delta_self, pipeline_name=src_clean.pipeline_name,
                                            perc_data_points=src_poisoned[const.KEY_PERC_DATA_POINTS],
                                            perc_features=src_poisoned[const.KEY_PERC_FEATURES])
                    delta_self = delta_self.rename(
                        lambda col: f'{const.PREFIX_DELTA}({col})' if col in columns_to_consider_in_delta_ else col,
                        axis='columns')
                    delta_self_results[k] = delta_self

                # now, we add the clean result at the beginning.
                model_qualities_vanilla[test_set_type].append(pd.concat([pd.DataFrame(results_of_model_vanilla.clean.model_quality[test_set_type]).T,
                                                   quality_vanilla_poisoned_df]).reset_index(drop=True))
                model_qualities_oracled[test_set_type].append(pd.concat([pd.DataFrame(results_of_model_oracled.clean.model_quality[test_set_type]).T,
                                                   quality_oracle_poisoned_df]).reset_index(drop=True))
                delta_self_vanilla[test_set_type].append(delta_self_results[VANILLA])
                delta_self_oracled[test_set_type].append(delta_self_results[ORACLE])


            # when we arrive here, we have all the results of an individual model (both vanilla and monolithic).
            # (Except for the delta_ref which we retrieve at the end due to how the function works).
            # we finally retrieve delta ref, which does not require to iterate over the test set type.
            raw_delta_ref = common.compute_delta_ref(first=results_of_model_vanilla,
                                                        baseline=results_of_model_oracled)
            for test_set_type, result in raw_delta_ref.items():
                delta_ref[test_set_type].append(result)

        for test_set_type in common.TestSetType:
            # now, we merge the different qualities. During this operation, we also rename the columns.
            # Note: we keep the separation between the different test sets
            model_quality_oracled_df_s[test_set_type] = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(
                model_qualities_oracled[test_set_type], pipeline_names=list(model_names))
            model_quality_vanilla_df_s[test_set_type] = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(
                model_qualities_vanilla[test_set_type], pipeline_names=list(model_names))
            delta_self_oracle_df_s[test_set_type] = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(
                delta_self_oracled[test_set_type], pipeline_names=list(model_names))
            delta_self_vanilla_df_s[test_set_type] = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(
                delta_self_vanilla[test_set_type], pipeline_names=list(model_names))
            delta_ref_vanilla_oracled_df_s[test_set_type] = utils_exp_post.merge_repeatedly_and_drop_unnecessary_columns(
                delta_ref[test_set_type], pipeline_names=list(model_names))

        return AnalyzedResultMonolithicModels(
            model_quality_vanilla=model_quality_vanilla_df_s, model_quality_oracle=model_quality_oracled_df_s,
            delta_self_vanilla=delta_self_vanilla_df_s, delta_self_oracle=delta_self_oracle_df_s,
            delta_ref_vanilla_oracle=delta_ref_vanilla_oracled_df_s)

    def export(self, config: ExportConfigBaseModels):
        if config.base_directory is None:
            return
        os.makedirs(config.base_directory, exist_ok=config.exists_ok)

        for target_name, src in [(base.FILE_NAME_EXPORT_MONO_VANILLA_QUALITY, self.model_quality_vanilla),
                                 (base.FILE_NAME_EXPORT_MONO_ORACLED_QUALITY, self.model_quality_oracle),
                                 (base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_SELF, self.delta_self_vanilla),
                                 (base.FILE_NAME_EXPORT_MONO_ORACLED_DELTA_SELF, self.delta_self_oracle),
                                 (base.FILE_NAME_EXPORT_MONO_VANILLA_DELTA_REF_AGAINST_MONO_ORACLED, self.delta_ref_vanilla_oracle),]:
            for test_set_type, df in src.items():
                df.to_csv(os.path.join(config.base_directory, f'{target_name}_{test_set_type.prefix()}.csv'), index=False)


class ExperimentMonolithicModels(base.AbstractExperiment, common.AbstractTrainModelWithRepMixin):

    def get_repetitions(self) -> int:
        return self.repetitions

    def get_X_y_test(self, test_set_type: TestSetType) -> typing.Tuple[np.ndarray, np.ndarray]:
        if test_set_type == TestSetType.CLEAN_TEST_SET:
            return self.X_test, self.y_test
        else:
            return self.X_train_clean, self.y_train_clean

    @property
    def analysis_class(self) -> typing.Type[AnalyzedResultMonolithicModels]:
        return AnalyzedResultMonolithicModels

    def __init__(self, X_train_clean: np.ndarray,
                 y_train_clean: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 monolithic_models: typing.Sequence[typing.Tuple[str, TEstimator]],
                 repetitions: int,
                 poisoned_datasets: xr.Dataset,
                 columns: typing.Optional[typing.List[str]] = None):
        super().__init__(repetitions=repetitions, clean_dataset_attrs={},
                         poisoned_datasets=poisoned_datasets, columns=columns)
        # check base models name.
        monolithic_models_name = set([b[0] for b in monolithic_models])
        if len(monolithic_models) != len(monolithic_models_name):
            raise ValueError(f'Some models have a duplicate names:\n{[b[0] for b in monolithic_models]}\n'
                             f'{monolithic_models_name}')

        self.X_train_clean = X_train_clean
        self.y_train_clean = y_train_clean
        self.X_test = X_test
        self.y_test = y_test
        self.monolithic_models = monolithic_models

    @staticmethod
    def from_dataset_generator(dg: dataset_generator.DatasetGenerator,
                               monolithic_models: typing.Sequence[typing.Tuple[str, TEstimator]],
                               repetitions: int):
        return ExperimentMonolithicModels(X_test=dg.X_test, y_test=dg.y_test, X_train_clean=dg.X_train_clean,
                                          y_train_clean=dg.y_train_clean, columns=dg.columns,
                                          poisoned_datasets=dg.all_datasets, repetitions=repetitions, monolithic_models=monolithic_models)

    def train_model_on_all_datasets(self, estimator: typing.Tuple[str, TEstimator]
                                    ) -> common.CleanPoisonedOutputPair[
        common.TrainSingleOutputMonolithicWithRep]:
        """
        One for each poisoned dataset.
        :param estimator:
        :return:
        """
        # first, train on clean dataset.
        pipeline_name = estimator[0]

        results_clean = self.train_model_with_rep(
            estimator=copy.deepcopy(estimator[1]), X_train=self.X_train_clean, y_train=self.y_train_clean,
            info=base.ExpInfo(perc_points=0.0, perc_features=0.0, pipeline_name=pipeline_name,),
            poisoning_info=np.zeros_like(self.y_train_clean))

        results_poisoned = joblib.Parallel(n_jobs=len(self.poisoned_datasets))(joblib.delayed(
            self.train_model_with_rep)(
            estimator=copy.deepcopy(estimator[1]),
            X_train=poisoned_dataset.sel(
                y=[val for val in poisoned_dataset.coords['y'].values
                   if val not in const.DG_IRRELEVANT_COLUMNS]).to_numpy(),
            # no need to reshape this value.
            y_train=poisoned_dataset.sel(
                y=const.COORD_LABEL
            ).to_numpy(),
            poisoning_info=poisoned_dataset.sel(y=const.COORD_POISONED).to_numpy(),
            info=base.ExpInfo(
                perc_points=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_POINTS],
                perc_features=poisoned_dataset.attrs[const.KEY_ATTR_POISONED][const.COORD_PERC_FEATURES],
                pipeline_name=estimator[0],
            ),
        ) for poisoned_dataset in self.poisoned_datasets.values())
        return common.CleanPoisonedOutputPair(clean=results_clean, poisoned=results_poisoned,
                                              pipeline_name=pipeline_name)

    def do(self
           ) -> typing.Tuple[typing.Sequence[typing.Sequence[common.TrainSingleOutputMonolithicWithRep]],
                typing.Sequence[typing.Sequence[common.TrainSingleOutputMonolithicWithRep]]]:
        result_vanilla = joblib.Parallel(n_jobs=len(self.monolithic_models))(
            joblib.delayed(self.train_model_on_all_datasets)(
                estimator=estimator
            ) for estimator in self.monolithic_models)
        result_oracled = joblib.Parallel(n_jobs=len(self.monolithic_models))(
            joblib.delayed(self.train_model_on_all_datasets)(
                estimator=(estimator[0], models.EstimatorWithOracle(wrapped=estimator[1]))
            ) for estimator in self.monolithic_models)
        return result_vanilla, result_oracled
