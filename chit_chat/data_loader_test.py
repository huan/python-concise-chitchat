"""
test
"""
from typing import (
    # Any,
    Iterable,
)

import numpy as np

import pytest                   # type: ignore
# pylint: disable=W0621

from .data_loader import DataLoader


@pytest.fixture(scope='module')
def loader() -> Iterable[DataLoader]:
    """ doc """
    loader = DataLoader()
    yield loader


def test_dataloader_smoke_testing(
        loader: DataLoader,
) -> None:
    """ doc """
    EXPECTED_BATCH_SIZE = 17
    left, _ = loader.get_batch(EXPECTED_BATCH_SIZE)
    assert len(left) == EXPECTED_BATCH_SIZE, 'should get back batch size'


def test_dataloader_batch_random(
        loader: DataLoader,
) -> None:
    '''doc'''
    queries1, _ = loader.get_batch()
    queries2, _ = loader.get_batch()

    not_equal = False
    for i, value in enumerate(queries1):
        if value != queries2[i]:
            not_equal = True

    assert not_equal, 'should different between the batches'


def test_dataloader_batch_type(
        loader: DataLoader,
) -> None:
    '''doc'''
    queries, _ = loader.get_batch()
    print(type(queries))
    assert isinstance(queries, np.ndarray), 'should get ndarray for queries'
    assert isinstance(queries[0], str),\
        'should get str type for item from queries'
