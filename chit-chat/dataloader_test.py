"""
test
"""
from typing import (
    # Any,
    Iterable,
)

import pytest                   # type: ignore
# pylint: disable=W0621

from dataloader import DataLoader


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


def test_dataloader_batch_random_and_type(
        loader: DataLoader,
) -> None:
    '''doc'''
    left, _ = loader.get_batch()
    sample = left[0]
    left, _ = loader.get_batch()
    assert isinstance(left, list), 'should get list for left sentence list'
    assert isinstance(sample, str), 'should get str type for sample item'
    assert sample != left[0]
