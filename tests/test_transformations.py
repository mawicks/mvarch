import torch

import pytest

from mvarch.deep_learning.transformations import (
    simple_get_portfolio_returns,
    proper_get_portfolio_returns,
)

ALLOCATION_BATCHES = 3
NUMBER_OF_SYMBOLS = 4
SEQUENCE_LENGTH = 7


def test_simple_get_portfolio_returns():
    allocations = torch.randn(ALLOCATION_BATCHES, NUMBER_OF_SYMBOLS)
    historical_returns = torch.randn(SEQUENCE_LENGTH, NUMBER_OF_SYMBOLS)
    result = simple_get_portfolio_returns(allocations, historical_returns)
    assert result.shape == (ALLOCATION_BATCHES, SEQUENCE_LENGTH)


def test_proper_get_portfolio_returns():
    allocations = torch.randn(ALLOCATION_BATCHES, NUMBER_OF_SYMBOLS)
    historical_returns = 0.01 * torch.randn(SEQUENCE_LENGTH, NUMBER_OF_SYMBOLS)

    comparison = simple_get_portfolio_returns(allocations, historical_returns)

    for reverse in (False, True):
        result = proper_get_portfolio_returns(
            allocations, historical_returns, reverse=reverse
        )

        assert result.shape == (ALLOCATION_BATCHES, SEQUENCE_LENGTH)

        error = torch.max(torch.abs(comparison - result))

        # The upper limit for the error hasn't been determined
        # rigorously.  It's just a check that the error is somewhat
        # small compared to the typical daily return. It should catch
        # gross errors, but certainly not all errors.  Also, there's no
        # check that the result is different for different values of
        # `reverse`.

        assert float(error) < 0.0005
