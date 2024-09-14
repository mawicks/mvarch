import torch

import pytest

from mvarch.deep_learning.transformations import (
    simple_get_portfolio_backwards_returns,
    proper_get_portfolio_backwards_returns,
)

ALLOCATION_BATCHES = 3
NUMBER_OF_SYMBOLS = 4
SEQUENCE_LENGTH = 7


def test_simple_portfolio_backwards_returns():
    allocations = torch.randn(ALLOCATION_BATCHES, NUMBER_OF_SYMBOLS)
    historical_returns = torch.randn(NUMBER_OF_SYMBOLS, SEQUENCE_LENGTH)
    result = simple_get_portfolio_backwards_returns(allocations, historical_returns)
    assert result.shape == (ALLOCATION_BATCHES, SEQUENCE_LENGTH)


def test_proper_portfolio_backward_returns():
    allocations = torch.randn(ALLOCATION_BATCHES, NUMBER_OF_SYMBOLS)
    historical_returns = 0.01 * torch.randn(NUMBER_OF_SYMBOLS, SEQUENCE_LENGTH)

    comparison = simple_get_portfolio_backwards_returns(allocations, historical_returns)
    result = proper_get_portfolio_backwards_returns(allocations, historical_returns)

    assert result.shape == (ALLOCATION_BATCHES, SEQUENCE_LENGTH)
