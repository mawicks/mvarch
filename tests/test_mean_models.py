import pytest

import torch

# Local modules
from mvarch.mean_models import MeanModel, ZeroMeanModel, ARMAMeanModel


def test_zero_mean_model():
    observations = torch.randn((10, 3))
    noise = torch.randn((10, 3))

    mean_model = ZeroMeanModel()
    mean_model.initialize_parameters(observations)

    parameters = mean_model.get_parameters()
    assert len(parameters) == 0

    mean_model.set_parameters(**{})

    optimizable_parameters = mean_model.get_optimizable_parameters()
    assert len(optimizable_parameters) == 0

    means, next_mean = mean_model.predict(observations)
    means, next_mean = mean_model._predict(observations, sample=False)
    means, next_mean = mean_model._predict(observations, sample=True)

    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    mean_model.log_parameters()


def test_ARMA_mean_model():
    observations = torch.randn((10, 3))
    noise = torch.randn((10, 3))

    mean_model = ARMAMeanModel()

    # Call several methods without having initializing parameters:
    with pytest.raises(RuntimeError):
        optimizable_parameters = mean_model.get_optimizable_parameters()
    with pytest.raises(RuntimeError):
        means, next_mean = mean_model._predict(observations, sample=False)

    mean_model.log_parameters()

    mean_model.initialize_parameters(observations)
    mean_model.log_parameters()

    optimizable_parameters = mean_model.get_optimizable_parameters()
    assert len(optimizable_parameters) == 4

    means, next_mean = mean_model.predict(observations)
    means, next_mean = mean_model._predict(observations, sample=False)
    means, next_mean = mean_model._predict(
        observations, sample=False, mean_initial_value=[0.001, 0.002, 0.003]
    )
    means, next_mean = mean_model._predict(observations, sample=True)
    means, next_mean = mean_model._predict(
        observations, sample=True, mean_initial_value=[0.001, 0.002, 0.003]
    )

    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    constructed_parameters = mean_model.get_parameters()
    assert len(constructed_parameters) == 5

    good_parameters = {
        "a": [0.8, 0.7, 0.6],
        "b": [0.1, 0.20, 0.3],
        "c": [0.1, 0.10, 0.3],
        "d": [1.0, 1.0, 1.0],
        "sample_mean": [0.001, -0.001, 0.005],
    }

    bad_parameters = {
        "a": [0.8, 0.7, 0.6],
        "b": [0.1, 0.20, 0.3, 0.4],
        "c": [0.1, 0.10, 0.3],
        "d": [1.0, 1.0],
        "sample_mean": [0.001, -0.001, 0.005],
    }
    mean_model.set_parameters(**good_parameters)

    with pytest.raises(ValueError):
        mean_model.set_parameters(**bad_parameters)
