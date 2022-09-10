import pytest

from mvarch.distributions import NormalDistribution, StudentTDistribution
from mvarch.univariate_models import UnivariateARCHModel, UnivariateUnitScalingModel
from mvarch.mean_models import ZeroMeanModel, ARMAMeanModel
from mvarch.parameters import TriangularParameter, FullParameter
from mvarch.model_factory import model_factory


def test_factory():
    with pytest.raises(ValueError):
        model = model_factory(distribution="xxx", mean="zero", univariate=None)

    # Case 1 - Defaults
    model = model_factory()
    assert type(model.univariate_model) == UnivariateARCHModel
    assert type(model.univariate_model.mean_model) == ZeroMeanModel
    assert type(model.distribution) == NormalDistribution
    assert model.parameter_type == FullParameter

    # Case 2
    model = model_factory(
        distribution="studentt",
        mean="arma",
        univariate="arch",
        constraint="triangular",
    )
    assert type(model.univariate_model) == UnivariateARCHModel
    assert type(model.univariate_model.mean_model) == ARMAMeanModel
    assert type(model.distribution) == StudentTDistribution
    assert model.parameter_type == TriangularParameter

    # Case 3 - Use None rather than a string
    model = model_factory(
        distribution="studentt",
        mean="arma",
        univariate=None,
        constraint="triangular",
    )
    assert type(model.univariate_model) == UnivariateUnitScalingModel
    assert type(model.univariate_model.mean_model) == ARMAMeanModel
    assert type(model.distribution) == StudentTDistribution
    assert model.parameter_type == TriangularParameter

    # Case 3 - No multivariate model
    model = model_factory(
        distribution="studentt",
        mean="arma",
        univariate="arch",
        multivariate=None,
    )
    assert type(model) == UnivariateARCHModel
    assert type(model.mean_model) == ARMAMeanModel
    assert type(model.distribution) == StudentTDistribution
