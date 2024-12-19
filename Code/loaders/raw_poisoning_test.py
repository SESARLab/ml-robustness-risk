import pytest
from sklearn import cluster

import poisoning
from . import raw_poisoning


@pytest.mark.parametrize('initial, expected', [
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
            poisoning.SelectorRandom()
    ),
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectionInfoLabelBiDirectionalMirrored'),
            poisoning.SelectionInfoLabelBiDirectionalMirrored()
    ),
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectorClustering'),
            poisoning.SelectorClustering()
    ),
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectorBoundary'),
            poisoning.SelectorBoundary()
    ),
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectorSCLFA'),
            poisoning.SelectorSCLFA()
    ),
    (
            raw_poisoning.SelectorRaw(name='__poisoning.SelectorClustering',
                                      init_kwargs={'distance_exp': 1,
                                                   'inner_algo_clazz': '_sklearn.cluster.KMeans'}),
            poisoning.SelectorClustering(distance_exp=1, inner_algo_clazz=cluster.KMeans)
    )
])
def test_selector(initial: raw_poisoning.SelectorRaw, expected):
    got = initial.parse()
    assert expected == got or type(expected) == type(got)


@pytest.mark.parametrize('initial, expected', [
    (
            raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
            poisoning.PerformerLabelFlippingMonoDirectional(),
    ),
    (
            raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingBiDirectional'),
            poisoning.PerformerLabelFlippingBiDirectional()
    )
])
def test_performer(initial: raw_poisoning.PerformerRaw, expected):
    got = initial.parse()
    assert expected == got


@pytest.mark.parametrize('initial, expected', [
    (
            raw_poisoning.PoisoningGenerationInfoRaw(
                selector=raw_poisoning.SelectorRaw(name='__poisoning.SelectorRandom'),
                performer=raw_poisoning.PerformerRaw(name='__poisoning.PerformerLabelFlippingMonoDirectional'),
                perc_data_points=[10, 15],
                perform_info_clazz='_poisoning.PerformInfoMonoDirectional',
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                selection_info_clazz='_poisoning.SelectionInfoLabelMonoDirectionalRandom',
                selection_info_kwargs={'from_label': 0}
            ),
            poisoning.PoisoningGenerationInput(
                selector=poisoning.SelectorRandom(),
                performer=poisoning.PerformerLabelFlippingMonoDirectional(),
                perc_data_points=[10, 15],
                selection_info_clazz=poisoning.SelectionInfoLabelMonoDirectionalRandom,
                selection_info_kwargs={'from_label': 0},
                perform_info_clazz=poisoning.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 0, 'to_label': 1}
            )
    )
])
def test_generation_input(initial: raw_poisoning.PoisoningGenerationInfoRaw,
                          expected: poisoning.PoisoningGenerationInput):
    got = initial.parse()
    sequence = got.generate_from_sequence()
    assert len(sequence) == 2
    assert len(sequence[0]) == len(expected.perc_data_points)
    assert len(sequence[1]) == len(expected.perc_data_points)

    for got_poisoning_algo in sequence[1]:
        assert type(expected.performer) == type(got_poisoning_algo.performer)
        assert type(expected.selector) == type(got_poisoning_algo.selector)
        assert expected.selection_info_clazz == type(got_poisoning_algo.selection_info)
        assert expected.perform_info_clazz == type(got_poisoning_algo.perform_info)
