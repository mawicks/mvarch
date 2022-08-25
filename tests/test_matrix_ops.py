import pytest

import torch

# Local modules
from mvarch import matrix_ops


@pytest.mark.parametrize(
    "m",
    [
        [[1, 0, 0], [-2, 1, 0], [3, -5, 1]],
        [[1, 0, 0], [-2, -1, 0], [3, 5, -1]],
        [[1, 0, 0], [-2, 0, 0], [3, 5, -1]],
        [[1, 0, 0], [-2, 0, 0], [3, -5, 1]],
    ],
)
def test_make_diagonal_nonnegative(m):
    """
    Test the function make_diagonal_nonnegative().
    When given a lower triangular matrix, it should
    return a lower triangular matrix with nonnegative diagonal
    entries which has the same product m @ m.T

    """
    m = torch.tril(torch.tensor(m))
    n = matrix_ops.make_diagonal_nonnegative(m)

    # Ensure that n is lower triangular.
    assert torch.all(n == torch.tril(n))

    # Ensure diagonal is nonnegative
    assert torch.all(torch.diag(n) >= 0.0)

    # Ensure "square" of the matrix is unchanged.
    mmT = m @ m.T
    nnT = n @ n.T
    assert torch.all(mmT == nnT)


def test_random_lower_triangular():
    """
    Test the function make_random_lower_triangular()
    which should return a random lower triangular matrix
    with entries uniformly distributed between -scale and scale.

    """
    for n in (1, 2, 3, 50):
        # The odds of the max not exceeding the threshhold should
        # be about a million to one.
        threshhold = -1 + 2 * 2 ** (-40 / n / (n + 1))

        for scale in (1.0, 2.0, 3.0):
            for requires_grad in (True, False):
                m = matrix_ops.random_lower_triangular(n, scale, requires_grad)

                # Confirm its dimension/shape
                assert m.shape == (n, n)

                # Confirm that it's lower triangular
                assert torch.all(m == torch.tril(m))

                # No entry should be larger than scale
                assert torch.max(torch.abs(m)) <= scale

                # But max value should be close to scale
                assert torch.max(m) > scale * threshhold

                # And min value should be close to -scale
                assert torch.min(m) < -scale * threshhold

                # requires_grad should be as specified
                assert m.requires_grad == requires_grad

                # Device isn't checked.  This test would
                # depend on hardware and isn't appropriate here.
