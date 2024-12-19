import typing

import pytest
from sklearn import datasets, preprocessing

from . import base, selectors


X, y = datasets.make_classification(500, 100)


@pytest.mark.parametrize('sel, info', [
    (selectors.SelectorClustering(scaling=preprocessing.MinMaxScaler()), base.SelectionInfoEmpty()),
    (selectors.SelectorRandom(), base.SelectionInfoEmpty()),
    (selectors.SelectorBoundary(), base.SelectionInfoEmpty()),
    (selectors.SelectorSCLFA(), base.SelectionInfoEmpty())
])
def test_selector(sel: selectors.AbstractSelector, info: typing.Optional[base.SelectionInfoLabelMonoDirectional]):
    sel.fit(X=X, y=y, selection_info=info)
    list_of_idx = sel.predict(X=X, y=y, selection_info=info)

    assert len(list_of_idx) == sel.output_size()
    if len(list_of_idx) == 2:
        assert len(list_of_idx[0]) + len(list_of_idx[1]) == len(X)
    else:
        assert len(list_of_idx[0]) <= len(X)
