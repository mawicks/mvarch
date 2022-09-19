import pytest

import numpy as np
import torch

# Local modules
from mvarch import distributions


@pytest.fixture()
def normal_distribution():
    return distributions.NormalDistribution()


@pytest.fixture()
def studentt_distribution():
    return distributions.StudentTDistribution()


def test_normal_distribution_set_get(normal_distribution):

    # Make sure log_parameters() can be called
    normal_distribution.log_parameters()

    # Make sure set_parameters() can be called
    normal_distribution.set_parameters()

    assert normal_distribution.std_dev() == 1.0

    # Make sure get_parameters() can be called and
    # returns an empty dict.
    parameters = normal_distribution.get_parameters()
    assert len(parameters) == 0

    # Make sure get_optimizable_parameters() can be called and
    # returns an empty list.
    optimizable_parameters = normal_distribution.get_optimizable_parameters()
    assert len(optimizable_parameters) == 0


def test_studentt_distribution_set_get(studentt_distribution):
    # Make sure log_parameters() can be called
    studentt_distribution.log_parameters()

    for df in (10, 20):
        studentt_distribution.set_device(None)  # For coverage only

        # Make sure set_parameters() can be called with a `df` keyword
        studentt_distribution.set_parameters(df=df)

        # Make sure get parmaeters() returns the df we just set.
        parameters = studentt_distribution.get_parameters()
        assert parameters["df"] == df


def test_normal_distribution_get_instance(normal_distribution):
    instance = normal_distribution.get_instance()

    # Here we're expecting a specific implementation: a PyTorch normal
    # distribution.
    assert isinstance(instance, torch.distributions.normal.Normal)

    # We require scale and loc to be zero so that this is a standard
    # normal distribution.  We control scale and center separately.
    assert instance.scale == 1.0
    assert instance.loc == 0.0


def test_studentt_distribution_get_instance(studentt_distribution):
    for df in (10, 20):
        studentt_distribution.set_parameters(df=df)
        # Get a new instance and make sure it has the correct df.

        instance = studentt_distribution.get_instance()
        # Here's we're expecting a very specific implementation: a PyTorch
        # StudentT distribution.
        assert isinstance(instance, torch.distributions.studentT.StudentT)
        assert float(instance.df) == df

        # Make sure the df we just set is the first optimizable parameter
        # with requires_grad set to True
        optimizable_parameters = studentt_distribution.get_optimizable_parameters()
        assert len(optimizable_parameters) == 1
        assert float(optimizable_parameters[0]) == df
        assert optimizable_parameters[0].requires_grad is True

        # We require scale and loc to be zero so this is a standard
        # distribution.  We control scale and center separately.
        assert instance.scale == 1.0
        assert instance.loc == 0.0


def test_normal_std_dev(normal_distribution):
    normal_distribution = distributions.NormalDistribution()
    assert normal_distribution.std_dev() == 1


def test_studentt_std_dev(studentt_distribution):
    studentt_distribution.set_parameters(df=3.0)
    assert studentt_distribution.std_dev() == float(
        torch.sqrt(torch.tensor(3.0, dtype=torch.float))
    )

    studentt_distribution.set_parameters(df=2.0)
    assert studentt_distribution.std_dev() == float("inf")

    studentt_distribution.set_parameters(df=1.0)
    assert np.isnan(studentt_distribution.std_dev())
