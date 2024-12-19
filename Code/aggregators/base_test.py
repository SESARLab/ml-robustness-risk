import numpy as np

import pipe


def help_test_aggregation(input_X, expected, p: pipe.ExtPipeline):
    result = p.fit_transform(X=input_X, y=None)
    # the output has two elements, X, and y
    # we are only interested in X, and we also ensure than
    # no values are set to y
    assert result[1] is None
    result = result[0]
    assert len(result) == len(expected)
    # assert np.all(result == expected)
    assert np.allclose(result, expected), f'{np.vstack([expected, result])}'
