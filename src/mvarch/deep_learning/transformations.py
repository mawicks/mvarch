import torch


def simple_get_portfolio_backwards_returns(allocations, historical_returns):
    """Use a simple linear formula to approximate the historical log-returns
    of a portfolio, given the historical log-returns of each asset in
    the portfolio.

    Arguments:
        historical_returns: shape of (number_of_symbols, sequence_size)
        allocation: shape of (allocation_batch_size, number_of_symbols)

    Returns:
        shape of (allocation_batch_size, sequence_size)

    """

    return torch.softmax(allocations, 1) @ historical_returns


def proper_get_portfolio_backwards_returns(
    allocations: torch.Tensor, historical_returns: torch.Tensor
):
    """Use a more complex formula to get the historical log-returns of a portfolio,
    given the historical log-returns of each asset int the portfolio.
    Assume the portfolio allocation is chosen today, then back-propogate
    the returns of each asset in the portfolio to get the historical
    asset values. Add then together in the value space rather then in
    the log-return space.

    Arguments:
        allocations: shape of (allocation_batch_size, number_of_symbols)
        historical_returns: shape of (number_of_symbols, sequence_size)

    Returns:
        shape of (allocation_batch_size, sequence_size)


    """
    allocation_batch_size, number_of_symbols = allocations.shape
    _, sequence_size = historical_returns.shape

    if _ != number_of_symbols:
        raise ValueError(
            f"Dimension 1 of allocations ({number_of_symbols}) must equal dimension 0 ({_}) of historical_returns"
        )

    log_allocation = torch.log_softmax(allocations, 1)
    # Accumulate the log returns in the reverse direction (flip the
    # order along the time dimension which is 1, and flip the sign)
    reversed_cumulative_log_returns = -torch.cumsum(
        torch.flip(historical_returns, [1]), 1
    )

    # The allocations are log allocations and the returns are log
    # returns, so each log allocation will be added to cumulative log
    # returns before exponentiating.  Make the allocations and
    # cumulative log returns conformal so they can simply be added.
    # Then we'll apply our favorite function logsumexp() so that
    # everything is always in log space and numerically stable.
    # The conformal dimension we'll use is (allocation_batch_size,
    # number_of_symbols, sequence_size).

    # First, expand the allocation along the time dimension (sequence_size)
    # Do this by calling unsqueeze() to get a time dimension, then
    # expand() the time dimension.
    desired_shape = (allocation_batch_size, number_of_symbols, sequence_size)

    log_allocation = log_allocation.unsqueeze(2).expand(*desired_shape)

    # Similiarly replicate the cumulative log returns for allocation
    # along the allocation batch (allocation_batch_size):

    reversed_cumulative_log_returns = reversed_cumulative_log_returns.unsqueeze(
        0
    ).expand(*desired_shape)

    cumulative_reversed_portfolio_log_returns = torch.logsumexp(
        log_allocation + reversed_cumulative_log_returns, dim=1
    )

    # Now flip again, append zeros, and diff:

    x = torch.diff(
        torch.concat(
            [
                torch.flip(cumulative_reversed_portfolio_log_returns, [1]),
                torch.zeros(allocation_batch_size, 1),
            ],
            dim=1,
        )
    )
    return x
