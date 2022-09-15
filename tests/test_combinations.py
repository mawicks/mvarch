import pytest

import itertools
import torch

from mvarch.distributions import NormalDistribution, StudentTDistribution
from mvarch.univariate_models import UnivariateARCHModel, UnivariateUnitScalingModel
from mvarch.mean_models import ZeroMeanModel, ARMAMeanModel
from mvarch.parameters import TriangularParameter, FullParameter
from mvarch.model_factory import model_factory

DISTRIBUTION_OPTIONS = ("normal", "studentt")
MEAN_OPTIONS = ("zero", "constant", "arma")
UNIVARIATE_OPTIONS = ("arch", "none")
MULTIVARIATE_OPTIONS = ("mvarch", "none")
CONSTRAINT_OPTIONS = ("scalar", "diagonal", "triangular", "none")


def test_combinations():
    """
    Run fit() on every possible combination of models and constraints to ensure
    there aren't hidden issues with specific combinations.  Use a small set of
    random data.  We're not testing for correctness here.  This test only ensures
    that these models train to completion.
    """
    observations = 0.01 * torch.rand(10, 3)

    for distribution, mean, univariate, multivariate, constraint in itertools.product(
        DISTRIBUTION_OPTIONS,
        MEAN_OPTIONS,
        UNIVARIATE_OPTIONS,
        MULTIVARIATE_OPTIONS,
        CONSTRAINT_OPTIONS,
    ):
        if univariate != "none" or multivariate != "none":
            model = model_factory(
                distribution=distribution,
                mean=mean,
                univariate=univariate,
                multivariate=multivariate,
                constraint=constraint,
            )
            model.fit(observations)
