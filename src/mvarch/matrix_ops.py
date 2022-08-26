# Standard Python
from typing import Union

# Common packages
import torch


@torch.no_grad()
def is_lower_triangular(m: torch.Tensor) -> bool:
    return bool(torch.all(torch.tril(m) == m))


def make_diagonal_nonnegative(m: torch.Tensor) -> torch.Tensor:
    """Given a single lower triangular matrix m, return an `equivalent` matrix
    having non-negative diagonal entries.  Here `equivalent` means m @ m.T is unchanged.

    Arguments:
        m: torch.Tensor of shape (n,n) that is lower triangular
    Returns:
        torch.Tensor: of shape (n, n) which is `equivalent` and has non-negative
        values on its diagonal
    """
    diag = torch.diag(m)
    diag_signs = torch.ones(diag.shape)
    diag_signs[diag < 0.0] = -1
    return m * diag_signs


def random_lower_triangular(
    n: int,
    scale: Union[torch.Tensor, float] = 1.0,
    requires_grad: bool = True,
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    """
    This function returns a random lower triangular matrix of
    dimension n by n, with entries uniformly distributed between
    -scale and scale.
    """
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device)
    m = torch.rand(n, n, device=device) - 0.5
    m = 2.0 * torch.abs(scale) * make_diagonal_nonnegative(torch.tril(m))
    m.requires_grad = requires_grad
    return m


if __name__ == "__main__":

    rlt = random_lower_triangular(3, 10.0)

    print(rlt)
