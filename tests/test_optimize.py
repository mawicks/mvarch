import pytest

import torch

# Local modules
import mvarch.optimize as optimize

# Define the parameters for a simple optimization problem.
# X_OPT will be the desired solution (minimizing X)
X_OPT = torch.tensor([1.0, 3.0], dtype=torch.float)

# A is an arbitrary matrix.
A = torch.tensor([[2.0, -1.0], [-4.0, 3.0]], dtype=torch.float)

# OPTIMAL_VALUE will be the minimum of the function
OPTIMAL_VALUE = 5.0

# EPS is the relative error for an acceptable solution.
EPS = 1e-6


def objective(x):
    # Construct a simple quadratic function that is minimized at x=X_OPT
    e = x - X_OPT
    z = A @ e
    return torch.sum(z * z) + OPTIMAL_VALUE


def test_optimize():
    """
    Test that the optimize()
    """
    # Initialize candidate x to vector of zeros.
    x = torch.zeros(2, requires_grad=True)

    optim = torch.optim.LBFGS(params=[x], lr=1.0)

    def closure():
        optim.zero_grad()
        loss = objective(x)
        loss.backward()
        return float(loss)

    # This is the function we're testing, which is just a wrapper for optim.step()
    optimize.optimize(optim, closure)

    optimum = objective(x)

    assert float(optimum - OPTIMAL_VALUE) <= EPS * OPTIMAL_VALUE
    e = x - X_OPT
    print(f"error: {e}")
    assert torch.norm(e) <= EPS * torch.norm(X_OPT)
