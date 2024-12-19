import tempfile

import pytest
from sklearn import datasets, model_selection

import const
import poisoning
from . import dataset_generator as dataset_generator


X, y = datasets.make_classification()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)


def get_dg(*, poisoning_generation_input: poisoning.PoisoningGenerationInput,
           generate: bool = True) -> dataset_generator.DatasetGenerator:
    dg = dataset_generator.DatasetGenerator.from_dataset_to_poison(
        poisoning_generation_input=poisoning_generation_input,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    if generate:
        dg.generate()
    return dg


@pytest.mark.parametrize('poisoning_generation_input', [
    (
            poisoning.PoisoningGenerationInput(
                perc_data_points=[10, 15, 20],
                performer=poisoning.PerformerLabelFlippingMonoDirectional(),
                selector=poisoning.SelectorRandom(),
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                perform_info_clazz=poisoning.PerformInfoMonoDirectional,
                # selection_info_clazz=poisoning.SelectionInfoLabelMonoDirectional,
                selection_info_clazz=poisoning.SelectionInfoEmpty,
                selection_info_kwargs={}
                # selection_info_kwargs={'from_label': 1, 'to_label': 0}
            )
    ),
    # (
    #     poisoning.PoisoningGenerationInput(
    #         perc_data_points=[10, 15],
    #         performer=poisoning.PerformerLabelFlippingMonoDirectional(),
    #         selector=
    #     )
    # )
])
def test_dataset_generator_no_override(poisoning_generation_input: poisoning.PoisoningGenerationInput):
    # modify the input to include column
    poisoning_generation_input.columns = [f'{i}' for i in range(X.shape[1])]

    dg = get_dg(poisoning_generation_input=poisoning_generation_input)

    assert len(dg.all_datasets) == len(poisoning_generation_input.perc_data_points)
    # now for each individual xr.DataArray check that it contains the expected coords
    # and shape
    expected_col = set(poisoning_generation_input.columns).union({const.COORD_POISONED, const.COORD_LABEL})
    for individual_poisoned_dataset_name, poisoning_algo in zip(
            dg.all_datasets.data_vars, dg.poisoning_algos):
        # + 2 because we have y and the column indicating if it is poisoned.
        assert dg.all_datasets[individual_poisoned_dataset_name].shape == (X_train.shape[0], X.shape[1] + 2)
        # check correctness of columns
        got_col = set(dg.all_datasets[individual_poisoned_dataset_name].coords['y'].values)
        assert expected_col == got_col
        # this is a bit hardcoded but ok.
        assert dg.all_datasets[individual_poisoned_dataset_name].attrs == {const.KEY_ATTR_POISONED: poisoning_algo.perform_info.get_info_as_dict()}


@pytest.mark.parametrize('poisoning_generation_input_pre, poisoning_generation_input_post', [
    (
            poisoning.PoisoningGenerationInput(
                perc_data_points=[10, 15, 20],
                performer=poisoning.PerformerLabelFlippingMonoDirectional(),
                selector=poisoning.SelectorRandom(),
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                perform_info_clazz=poisoning.PerformInfoMonoDirectional,
                selection_info_clazz=poisoning.SelectionInfoEmpty,
                selection_info_kwargs={}
            ),
            poisoning.PoisoningGenerationInput(
                perc_data_points=[10, 15],
                performer=poisoning.PerformerLabelFlippingMonoDirectional(),
                selector=poisoning.SelectorRandom(),
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                perform_info_clazz=poisoning.PerformInfoMonoDirectional,
                selection_info_clazz=poisoning.SelectionInfoEmpty,
                selection_info_kwargs={}
            ),
    )
])
def test_import_with_smaller(poisoning_generation_input_pre: poisoning.PoisoningGenerationInput,
                             poisoning_generation_input_post: poisoning.PoisoningGenerationInput):

    # modify the input to include column
    poisoning_generation_input_pre.columns = [f'{i}' for i in range(X.shape[1])]
    poisoning_generation_input_post.columns = [f'{i}' for i in range(X.shape[1])]

    dg = get_dg(poisoning_generation_input=poisoning_generation_input_pre)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dg.export(base_directory=tmp_dir, exists_ok=True)

        # now, we re-import it.
        dg_imported = dataset_generator.DatasetGenerator.import_from_directory(
            base_directory=tmp_dir, poisoning_generation_input=poisoning_generation_input_post)

    # now, the length should be smaller.
    points, wrappers = poisoning_generation_input_post.generate_from_sequence()
    assert len(dg_imported) == len(points)
