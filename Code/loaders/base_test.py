import types

import pytest
import sklearn

import assignments
import experiments
from . import base
import poisoning


@pytest.mark.parametrize('name, expected', [
    (
            'poisoning.PerformerLabelFlippingMonoDirectional',
            poisoning.PerformerLabelFlippingMonoDirectional
    )
])
def test_import_if_needed(name: str, expected):
    got = base.import_if_needed(name)
    assert got == expected


@pytest.mark.parametrize('initial, expected', [
    (
            # this shall not be changed
            {'a': 1}, {'a': 1}
    ),
    (
            # here we expect a "type loading" i.e., no instantiation
            {'a': 1, 'model': '_sklearn.tree.DecisionTreeClassifier'},
            {'a': 1, 'model': sklearn.tree.DecisionTreeClassifier}
    ),
    (
            # here we expect a "basic class loading" where we instantiate a class
            # with no arguments.
            {'model': '__sklearn.utils.Bunch'}, {'model': sklearn.utils.Bunch()}
    ),
    (
            # here we expect the most sophisticated loading,
            # where we instantiate a class with some arguments.
            {'model': '__experiments.ExportConfigIoP', 'model_kwargs__': {'exists_ok': True}},
            {'model': experiments.ExportConfigIoP(exists_ok=True)}
    ),
    (
            # here instead we test that when kwargs does not end with '__'
            # it is kept as is.
            {'model': '__experiments.ExportConfigIoP', 'model_kwargs': {'exists_ok': True}},
            {'model': experiments.ExportConfigIoP(), 'model_kwargs': {'exists_ok': True}}
    ),
    (
            # here we load an enum which is tricky.
            {'how': '_assignments.SingleValuedRouterType.MODULO'},
            {'how': assignments.SingleValuedRouterType.MODULO}
    ),
    # (
    #         # here we test the loading of a code
    #         {'how': '_assignments.SingleValuedRouterType.MODULO',
    #          '___code': 'lambda a: a'},
    #         {'how': assignments.SingleValuedRouterType.MODULO,
    #          'code': lambda a: a}
    # )
])
def test_load_kwargs(initial: dict, expected: dict):
    got = base.fill_kwargs(initial)
    assert set(got.keys()) == set(expected.keys())
    # a little more detailed check because lambdas cannot be checked directly.
    for k in expected.keys():
        if type(expected[k]) is not types.FunctionType and type(expected[k]) is not types.LambdaType:
            assert expected[k] == got[k]
        else:
            assert type(expected[k]) is types.FunctionType or type(expected[k]) is types.LambdaType


@pytest.mark.parametrize('name, func_kwargs, expected', [
    (
            '__sklearn.utils.Bunch', None, sklearn.utils.Bunch()
    ),
    (
            '__sklearn.utils.Bunch', {'a': 1, 'b': 2}, sklearn.utils.Bunch(a=1, b=2)
    ),
    (
            '__sklearn.utils.Bunch', {'a': '__sklearn.utils.Bunch'}, sklearn.utils.Bunch(a=sklearn.utils.Bunch())
    ),
    (
            # a more complex example that works :)
            '__sklearn.utils.Bunch',
            {'a': '__sklearn.utils.Bunch', 'a_kwargs__': {'b': 2}},
            sklearn.utils.Bunch(a=sklearn.utils.Bunch(b=2))
    )
])
def test_load_func(name, func_kwargs, expected):
    got = base.load_func(name, func_kwargs)
    assert got == expected
