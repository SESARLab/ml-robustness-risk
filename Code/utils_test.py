import pandas as pd
import pytest

import utils


@pytest.mark.parametrize('dfs, mask, on', [
    (
            [
                pd.DataFrame(data=[[11, 1, 2, 3], [12, 4, 5, 6]], columns=['I', 'A', 'B', 'C']),
                pd.DataFrame(data=[[11, 10, 20, 30], [12, 40, 50, 60]], columns=['I', 'A', 'B', 'C']),
                pd.DataFrame(data=[[11, 100, 200, 300], [12, 400, 500, 600]], columns=['I', 'A', 'B', 'C']),
            ],
            [(False, '_1'), (True, '2_'), (False, '_3')],
            'I'
    )
])
def test_merge_multiple(dfs, mask, on):
    result = utils.merge_multiple(dfs=dfs, mask=mask, on=on)
    # - ((len(dfs)-1) * len_on) to account for
    # the column(s) that are kept during join.
    len_on = len(on) if isinstance(on, list) else 1
    assert result.shape[1] == sum(df.shape[1] for df in dfs) - ((len(dfs) - 1) * len_on)
