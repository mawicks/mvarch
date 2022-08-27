import pytest

import torch

# Local modules
from mvarch import distributions


def test_normal_distribution():
    distribution = distributions.NormalDistribution()

    # Make sure log_parameters() can be called
    distribution.log_parameters()

    # Make sure set_parameters() can be called
    distribution.set_parameters()

    # Make sure get_parameters() can be called and
    # returns an empty dict.
    parameters = distribution.get_parameters()
    assert len(parameters) == 0

    # Make sure get_optimizable_parameters() can be called and
    # returns an empty list.
    optimizable_parameters = distribution.get_optimizable_parameters()
    assert len(optimizable_parameters) == 0

    instance = distribution.get_instance()

    # Here we're expecting a specific implementation: a PyTorch normal
    # distribution.
    assert isinstance(instance, torch.distributions.normal.Normal)

    # We require scale and loc to be zero so that this is a standard
    # normal distribution.  We control scale and center separately.
    assert instance.scale == 1.0
    assert instance.loc == 0.0


def test_studentt_distribution():
    distribution = distributions.StudentTDistribution()

    # Make sure log_parameters() can be called
    distribution.log_parameters()

    for df in (10, 20):
        distribution.set_device(None)  # For coverage only

        # Make sure set_parameters() can be called with a `df` keyword
        distribution.set_parameters(df=df)

        # Make sure get parmaeters() returns the df we just set.
        parameters = distribution.get_parameters()
        assert parameters["df"] == df

        # Get a new instance and make sure it has the correct df.
        instance = distribution.get_instance()
        # Here's we're expecting a very specific implementation: a PyTorch
        # StudentT distribution.
        assert isinstance(instance, torch.distributions.studentT.StudentT)
        assert float(instance.df) == df

        # Make sure the df we just set is the first optimizable parameter
        # with requires_grad set to True
        optimizable_parameters = distribution.get_optimizable_parameters()
        assert len(optimizable_parameters) == 1
        assert float(optimizable_parameters[0]) == df
        assert optimizable_parameters[0].requires_grad is True

        # We require scale and loc to be zero so this is a standard
        # distribution.  We control scale and center separately.
        assert instance.scale == 1.0
        assert instance.loc == 0.0
