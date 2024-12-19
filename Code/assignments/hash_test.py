import numpy as np
import pytest
from sklearn import pipeline

from .hash import SingleValuedRouter, Hasher

test_data_fibo = [(0, 0),
                  (1, 19),
                  (2, 7),
                  (3, 27),
                  (4, 15),
                  (5, 2),
                  (6, 22),
                  (7, 10),
                  (8, 30),
                  (9, 17),
                  (10, 5),
                  (11, 25),
                  (12, 13),
                  (13, 1),
                  (14, 20),
                  (15, 8),
                  (16, 28),
                  (17, 16),
                  (18, 3),
                  (19, 23),
                  (20, 11),
                  (21, 31),
                  (22, 19),
                  (23, 6),
                  (24, 26),
                  (25, 14),
                  (26, 2),
                  (27, 21),
                  (28, 9),
                  (30, 17),
                  (31, 5)]


def test_partition_fibonacci():
    p = pipeline.make_pipeline(SingleValuedRouter(algo='fibonacci', N=32))
    x = np.array([data[0] for data in test_data_fibo])
    y = p.fit(x).transform(x)
    expected_y = np.array([data[1] for data in test_data_fibo])
    assert np.all(expected_y == y)


@pytest.mark.parametrize('algo_kwargs', [
    {'algo': 'md5'},
    {'algo': 'sha3_224'},
    {'algo': 'siphash'},
])
def test_partition_hash(algo_kwargs):
    p = Hasher(**algo_kwargs)
    x = np.random.rand(10, 5)
    hashed = p.fit(x).transform(x)
    assert len(hashed) == len(x)

# def test_partition_count(algo):
#     p = pipeline.make_pipeline(algo)
#     x = np.random.default_rng().integers(low=0, high=32, size=(32,))
#     p.fit(x).transform(x)
