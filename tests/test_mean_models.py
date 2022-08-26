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

    mean_model._predict(observations, sample=False)
    mean_model._predict(observations, sample=True)

    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    mean_model.log_parameters()


def test_ARMA_mean_model():
    observations = torch.randn((10, 3))
    noise = torch.randn((10, 3))

    mean_model = ARMAMeanModel()
    mean_model.initialize_parameters(observations)

    optimizable_parameters = mean_model.get_optimizable_parameters()
    assert len(optimizable_parameters) == 4

    mean_model._predict(observations, sample=False)
    mean_model._predict(observations, sample=True)

    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    parameters = mean_model.get_parameters()
    assert len(parameters) == 5

    mean_model.log_parameters()


parameterss = {
    "a": [0.8, 0.7, 0.6],
    "b": [0.1, 0.20, 0.3],
    "c": [0.1, 0.10, 0.3],
    "d": [1.0, 1.0, 1.0],
}
