import filecmp
import itertools
import os
import tempfile
import typing

import numpy as np
import pytest
from sklearn import datasets, model_selection

from . import base, raw_dataset, raw_poisoning, raw_exp_iop, raw_exp_ensemble_plain_advanced, raw_exp_ensemble_risk, \
    raw_exp_monolithic_models
from experiments import dataset_generator_test as dg_test
import const
import experiments
import poisoning


def X_y_to_csv(X, y, path):
    arr = np.hstack([X, y.reshape(-1, 1)])
    np.savetxt(path, arr, delimiter=',',
               header=','.join([f'col{i}' for i in range(X.shape[1])] + [const.COORD_LABEL]))


def export_and_load_dg(
        initial: raw_dataset.DatasetToPoisonRaw | raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw | raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw | raw_exp_monolithic_models.ExperimentMonolithicModelRaw | raw_exp_iop.ExperimentIoPRaw,
        X=None, y=None):
    if X is None or y is None:
        X, y = datasets.make_classification(100, 10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    with tempfile.NamedTemporaryFile() as training_file, tempfile.NamedTemporaryFile() as testing_file:
        X_y_to_csv(X_train, y_train, training_file.name)
        X_y_to_csv(X_test, y_test, testing_file.name)

        if isinstance(initial, raw_dataset.DatasetToPoisonRaw):

            initial.dataset_path_training = training_file.name
            initial.dataset_path_testing = testing_file.name

        else:

            assert isinstance(initial,
                              (raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw,
                               raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw,
                               raw_exp_monolithic_models.ExperimentMonolithicModelRaw,
                               raw_exp_iop.ExperimentIoPRaw))

            initial.dataset_config.dataset_path_training = training_file.name
            initial.dataset_config.dataset_path_testing = testing_file.name

        with tempfile.TemporaryDirectory() as temp_dir:
            if isinstance(initial, raw_dataset.DatasetToPoisonRaw):
                target_func = initial.parse_and_load
                target_func_kwargs = {'base_output_directory': temp_dir, 'exists_ok': True}
            else:
                initial.base_output_directory = temp_dir
                target_func = initial.parse
                target_func_kwargs = {}
            # got = initial.parse_and_load(base_output_directory=temp_dir, exists_ok=True)
            got = target_func(**target_func_kwargs)

            files = os.listdir(temp_dir)
            assert files == [base.BASE_OUTPUT_DIR_DATASET]
    return got


def check_arrays_from_got(X_test, X_train, got, y_test, y_train):
    for got_arr, expected_arr, msg in [
        (got.X_train_clean, X_train, 'X_train_clean'),
        (got.y_train_clean, y_train, 'y_train_clean'),
        (got.X_test, X_test, 'X_test'),
        (got.y_test, y_test, 'y_test'),
    ]:
        assert np.all(got_arr == expected_arr), f'Mismatch: {msg}'


@pytest.mark.parametrize('initial, expected_base', [
    (
            raw_dataset.DatasetToPoisonRaw(
                dataset_path_training='',
                dataset_path_testing='',
                poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
                    selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                    performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                    perc_data_points=[10, 15],
                    perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                    perform_info_kwargs={'from_label': 0, 'to_label': 1},
                    selection_info_clazz='_poisoning.SelectionInfoLabelMonoDirectional',
                    selection_info_kwargs={'from_label': 0, 'to_label': 1}
                )),
            poisoning.PoisoningGenerationInput(
                selector=poisoning.SelectorRandom(),
                performer=poisoning.PerformerLabelFlippingMonoDirectional(),
                perc_data_points=[10, 15],
                selection_info_clazz=poisoning.SelectionInfoLabelMonoDirectional,
                selection_info_kwargs={'from_label': 0, 'to_label': 1},
                perform_info_clazz=poisoning.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 0, 'to_label': 1}
            )
    ),
])
def test_parse_to_poison(initial: raw_dataset.DatasetToPoisonRaw, expected_base: poisoning.PoisoningGenerationInput):
    # we need to create a temp dir where we export the dataset
    X, y = datasets.make_classification(100, 10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=False)

    with tempfile.NamedTemporaryFile() as temp_file_training:
        X_y_to_csv(X_train, y_train, temp_file_training.name)

        with tempfile.NamedTemporaryFile() as temp_file_testing:
            X_y_to_csv(X_test, y_test, temp_file_testing.name)

            initial.dataset_path_training = temp_file_training.name
            initial.dataset_path_testing = temp_file_testing.name

            got = initial.parse()

    check_arrays_from_got(X_test, X_train, got, y_test, y_train)

    assert len(got.poisoning_algos) == len(initial.poisoning_input.perc_data_points)


@pytest.mark.parametrize('initial', [
    raw_dataset.DatasetToPoisonRaw(
        dataset_path_training='',
        dataset_path_testing='',
        poisoning_input=raw_poisoning.PoisoningGenerationInfoRaw(
            selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
            performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
            perc_data_points=[5, 10, 15],
            perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
            perform_info_kwargs={'from_label': 0, 'to_label': 1},
            selection_info_clazz='_poisoning.SelectionInfoEmpty',
            selection_info_kwargs={},
        )),
])
def test_parse_and_load_to_poison(initial: raw_dataset.DatasetToPoisonRaw):
    X, y = datasets.make_classification(100, 10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=False)
    with tempfile.NamedTemporaryFile() as temp_file_training:
        X_y_to_csv(X_train, y_train, temp_file_training.name)

        with tempfile.NamedTemporaryFile() as temp_file_testing:
            X_y_to_csv(X_test, y_test, temp_file_testing.name)

            initial.dataset_path_training = temp_file_training.name
            initial.dataset_path_testing = temp_file_testing.name

            with tempfile.TemporaryDirectory() as temp_dir:
                got = initial.parse_and_load(base_output_directory=temp_dir, exists_ok=True)

                check_arrays_from_got(X_test=X_test, X_train=X_train, got=got, y_test=y_test, y_train=y_train)
                assert len(got.poisoning_algos) == len(initial.poisoning_input.perc_data_points)

                assert len(got) == len(initial.poisoning_input.perc_data_points)
                for got_poisoning_algo in got.poisoning_algos:
                    assert initial.poisoning_input.selector.name.removeprefix('__').split('.')[-1] in \
                           str(type(got_poisoning_algo.selector)).split('.')[-1]
                    assert initial.poisoning_input.selector.name.removeprefix('__').split('.')[-1] in \
                           str(type(got_poisoning_algo.selector)).split('.')[-1]

                # we also check that stuff has been exported
                sub_dirs = os.listdir(temp_dir)
                assert set(sub_dirs) == {experiments.DIR_DATASET_NAME_EXPORT_BINARY,
                                         experiments.DIR_DATASET_NAME_EXPORT_CSV}
                sub_binary = os.listdir(os.path.join(temp_dir, experiments.DIR_DATASET_NAME_EXPORT_BINARY))
                # clean, test, poisoned
                assert len(sub_binary) == 3
                sub_csv = os.listdir(os.path.join(temp_dir, experiments.DIR_DATASET_NAME_EXPORT_CSV))
                # clean, test, |perc_point|
                assert len(sub_csv) == 1 + 1 + len(initial.poisoning_input.perc_data_points)


@pytest.mark.parametrize('poisoning_input_raw', [
    raw_poisoning.PoisoningGenerationInfoRaw(
        selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
        performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
        perc_data_points=[10, 15],
        perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
        perform_info_kwargs={'from_label': 0, 'to_label': 1},
        selection_info_clazz='_poisoning.SelectionInfoEmpty',
        selection_info_kwargs={}
    )
])
def test_parse_already_existing(poisoning_input_raw: raw_poisoning.PoisoningGenerationInfoRaw):
    # being already poisoning, we need to create it :)
    poisoning_input = poisoning_input_raw.parse()
    # if we shuffle X and y it then becomes quite difficult to compare stuff later.
    X, y = datasets.make_classification(100, 10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=False)

    dg = experiments.DatasetGenerator.from_dataset_to_poison(X_train=X_train, X_test=X_test,
                                                             y_train=y_train, y_test=y_test,
                                                             poisoning_generation_input=poisoning_input)
    dg.generate()
    with tempfile.TemporaryDirectory() as temp_dir:
        dg.export(base_directory=temp_dir, exists_ok=True)

        initial = raw_dataset.DatasetAlreadyPoisonedRaw(exported_dataset_path=temp_dir,
                                                        poisoning_input=poisoning_input_raw)

        # ok, now we import it.
        got = initial.parse()

        check_arrays_from_got(X_test=X_test, X_train=X_train, got=got, y_test=y_test, y_train=y_train)

        assert len(got.poisoning_algos) == len(poisoning_input.generate_from_sequence())
        assert len(dg) == len(poisoning_input.generate_from_sequence())
        assert np.all(dg.all_datasets == got.all_datasets)
        assert set(dg.all_datasets.keys()) == set(got.all_datasets.keys())
        for k in dg.all_datasets.keys():
            assert dg.all_datasets[k].attrs == got.all_datasets[k].attrs


@pytest.mark.parametrize('poisoning_input_raw', [
    raw_poisoning.PoisoningGenerationInfoRaw(
        selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
        performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
        perc_data_points=[5.0, 10.0, 15.0],
        perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
        perform_info_kwargs={'from_label': 0, 'to_label': 1},
        selection_info_clazz='_poisoning.SelectionInfoEmpty',
        selection_info_kwargs={}
    )
])
def test_parse_and_load_already_poisoned(poisoning_input_raw: raw_poisoning.PoisoningGenerationInfoRaw):
    def _ls_with_abs_path(base_path) -> typing.Iterable[str]:
        return map(lambda f: os.path.join(base_path, f), os.listdir(base_path))

    poisoning_input = poisoning_input_raw.parse()

    # X, y = datasets.make_classification(100, 10)
    # dg = experiments.DatasetGenerator.from_dataset_to_poison(X, y, poisoning_generation_input=poisoning_input)
    # dg.generate()
    dg = dg_test.get_dg(poisoning_generation_input=poisoning_input)

    with tempfile.TemporaryDirectory() as temp_dir_1:
        dg.export(base_directory=temp_dir_1, exists_ok=True)

        initial = raw_dataset.DatasetAlreadyPoisonedRaw(exported_dataset_path=temp_dir_1,
                                                        poisoning_input=poisoning_input_raw)

        files_csv_temp1 = _ls_with_abs_path(os.path.join(temp_dir_1, experiments.DIR_DATASET_NAME_EXPORT_CSV))
        files_binary_temp1 = _ls_with_abs_path(os.path.join(temp_dir_1, experiments.DIR_DATASET_NAME_EXPORT_BINARY))

        with tempfile.TemporaryDirectory() as temp_dir_2:
            got = initial.parse_and_load(base_output_directory=temp_dir_2, exists_ok=True)

            # now here ensure that temp_dir_2 is empty (no re-generation).
            assert len(os.listdir(temp_dir_2)) == 0

        # now here to check correctness of import, we export the dataset to another directory,
        # and we compare it against the imported one.
        with tempfile.TemporaryDirectory() as temp_dir_2:
            got.export(base_directory=temp_dir_2, exists_ok=True)

            # here, we test the equality of the files directly.
            # we therefore create pairs of pairs to check.
            # note that we can't use dircmp because it uses a shallow check
            # looking at file attributes only.

            files_csv_temp2 = _ls_with_abs_path(os.path.join(temp_dir_2, experiments.DIR_DATASET_NAME_EXPORT_CSV))
            files_binary_temp2 = _ls_with_abs_path(os.path.join(temp_dir_2, experiments.DIR_DATASET_NAME_EXPORT_BINARY))

            for f1, f2 in zip(itertools.chain(files_csv_temp1, files_binary_temp1),
                              itertools.chain(files_csv_temp2, files_binary_temp2)):
                assert filecmp.cmp(f1, f2, shallow=False)
