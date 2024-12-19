import typing

import pytest

from . import base, generator, selectors, performers, wrapper

TSelector = typing.TypeVar('TSelector', bound=selectors.AbstractSelector)
TPerformer = typing.TypeVar('TPerformer', bound=performers.AbstractPerformer)


@pytest.mark.parametrize('gen_input, expected', [
    (
            generator.PoisoningGenerationInput(
                perc_data_points=[2.5, 5.0],
                perform_info_clazz=base.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 0, 'to_label': 1},
                performer=performers.PerformerLabelFlippingMonoDirectional(),
                selection_info_clazz=base.SelectionInfoEmpty,
                selection_info_kwargs={},
                selector=selectors.SelectorRandom()
            ),
            (
                    [(2.5, 0.0), (5.0, 0.0)],
                    [
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorRandom(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=2.5,
                                                                                       from_label=0,
                                                                                       to_label=1),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorRandom(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=5.0,
                                                                                       from_label=0,
                                                                                       to_label=1),
                                          performer=performers.PerformerLabelFlippingMonoDirectional())
                    ]
            )
    ),
    (
            generator.PoisoningGenerationInput(
                perc_data_points=[2.5, 5.5],
                perform_info_clazz=base.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                performer=performers.PerformerLabelFlippingMonoDirectional(),
                selection_info_clazz=base.SelectionInfoEmpty,
                selection_info_kwargs={},
                selector=selectors.SelectorClustering()
            ),
            (
                    [(2.5, 0.0), (5.5, 0.0)],
                    [
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorClustering(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=2.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorClustering(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=5.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional())
                    ]
            )
    ),
    (
            generator.PoisoningGenerationInput(
                perc_data_points=[2.5, 5.5, 6.5],
                perform_info_clazz=base.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                performer=performers.PerformerLabelFlippingMonoDirectional(),
                selection_info_clazz=base.SelectionInfoEmpty,
                selection_info_kwargs={},
                selector=selectors.SelectorBoundary()
            ),
            (
                    [(2.5, 0.0), (5.5, 0.0), (6.5, 0.0)],
                    [
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorBoundary(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=2.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorBoundary(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=5.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorBoundary(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=6.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional())
                    ]
            )
    ),
    (
            generator.PoisoningGenerationInput(
                perc_data_points=[2.5, 5.5],
                perform_info_clazz=base.PerformInfoMonoDirectional,
                perform_info_kwargs={'from_label': 1, 'to_label': 0},
                performer=performers.PerformerLabelFlippingMonoDirectional(),
                selection_info_clazz=base.SelectionInfoEmpty,
                selection_info_kwargs={},
                selector=selectors.SelectorSCLFA()
            ),
            (
                    [(2.5, 0.0), (5.5, 0.0)],
                    [
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorSCLFA(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=2.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorSCLFA(),
                                          perform_info=base.PerformInfoMonoDirectional(perc_data_points=5.5,
                                                                                       from_label=1,
                                                                                       to_label=0),
                                          performer=performers.PerformerLabelFlippingMonoDirectional()),
                    ]
            )
    ),
    (
            generator.PoisoningGenerationInput(
                perc_data_points=[2.5, 5.6],
                perform_info_clazz=base.PerformInfoBiDirectionalMirrored,
                perform_info_kwargs={'label_a': 1, 'label_b': 0},
                performer=performers.PerformerLabelFlippingBiDirectional(),
                selection_info_clazz=base.SelectionInfoEmpty,
                selection_info_kwargs={},
                selector=selectors.SelectorSCLFA()
            ),
            (
                    [(2.5, 0.0), (5.6, 0.0)],
                    [
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorSCLFA(),
                                          perform_info=base.PerformInfoBiDirectionalMirrored(
                                              perc_data_points=2.5,
                                              label_a=1,
                                              label_b=0),
                                          performer=performers.PerformerLabelFlippingBiDirectional()),
                        wrapper.Poisoning(selection_info=base.SelectionInfoEmpty(),
                                          selector=selectors.SelectorSCLFA(),
                                          perform_info=base.PerformInfoBiDirectionalMirrored(
                                              perc_data_points=5.6,
                                              label_a=1,
                                              label_b=0),
                                          performer=performers.PerformerLabelFlippingBiDirectional()),
                    ]
            )
    )
])
def test_generate(gen_input: generator.PoisoningGenerationInput,
                  expected: typing.Tuple[..., typing.List[wrapper.Poisoning]]):
    got = gen_input.generate_from_sequence()

    poisoning_points = got[0]
    wrappers = got[1]

    assert len(poisoning_points) == len(wrappers)

    for got_points, expected_points in zip(poisoning_points, expected[0]):
        assert got_points[0] == expected_points[0]
        assert got_points[1] == expected_points[1]

    for got_wrapper, expected_wrapper in zip(wrappers, expected[1]):
        assert type(got_wrapper.selector) == type(expected_wrapper.selector)
        assert type(got_wrapper.performer) == type(expected_wrapper.performer)
        assert got_wrapper.perform_info == expected_wrapper.perform_info
        assert got_wrapper.selection_info == expected_wrapper.selection_info
