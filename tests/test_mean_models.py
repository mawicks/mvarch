import pytest

import torch

# Local modules
from mvarch.mean_models import ZeroMeanModel, ConstantMeanModel, ARMAMeanModel
from . import utils


def check_constant_prediction(model, observations, constant_mean_value):
    expanded_constant_mean_value = constant_mean_value.unsqueeze(0).expand(
        observations.shape
    )

    means, next_mean = model.predict(observations)
    assert torch.all(means == expanded_constant_mean_value)
    assert torch.all(next_mean == constant_mean_value)

    means, next_mean = model._predict(observations, sample=False)
    assert torch.all(means == expanded_constant_mean_value)
    assert torch.all(next_mean == constant_mean_value)

    means, next_mean = model._predict(observations, sample=True)
    assert torch.all(means == expanded_constant_mean_value)
    assert torch.all(next_mean == constant_mean_value)


def test_zero_mean_model():
    observations = torch.randn((10, 3))
    noise = torch.randn(observations.shape)

    mean_model = ZeroMeanModel()

    utils.set_and_check_parameters(mean_model, observations, {}, 0, 0)

    check_constant_prediction(
        mean_model, observations, torch.zeros(observations.shape[1])
    )

    # Calling sample() on a mean model isn't allowed.
    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    mean_model.log_parameters()


CONSTANT_MEAN_VALUE = {"mu": [0.001, 0.002, 0.003]}


def test_constant_mean_model():
    constant_mean_value = torch.tensor(CONSTANT_MEAN_VALUE["mu"])
    observations = torch.randn((10, 3))
    noise = torch.randn(observations.shape)

    mean_model = ConstantMeanModel()

    with pytest.raises(RuntimeError):
        mean_model.get_optimizable_parameters()

    with pytest.raises(RuntimeError):
        mean_model._predict(observations, sample=False)

    utils.set_and_check_parameters(mean_model, observations, CONSTANT_MEAN_VALUE, 1, 1)

    check_constant_prediction(mean_model, observations, constant_mean_value)

    # Calling sample() on a mean model isn't allowed.
    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    mean_model.log_parameters()


ARMA_VALID_PARAMETERS = {
    "a": [0.8, 0.7],
    "b": [0.1, 0.2],
    "c": [0.03, 0.07],
    "d": [2.0, 2.0],
    "sample_mean": [0.001, -0.002],
}
ARMA_MEAN_INITIAL_VALUE = [0.001, 0.002]
DEFAULT_INITIAL_VALUE = [0.002, -0.004]  # This is d * sample_mean
ARMA_OBSERVATIONS = [
    [0.003, -0.005],
    [0.0, 0.0],
]

# Here's what should happen:
#    mu0 = [.001, 0.002]
#    mu1 = a*initial_value + b*obs0 + c*sample_mean = [.00113, .0003]
#    mu2 = a*mu1 + b*obs1 + c*sample_mean = [9.34e-4, 4.2e-5]
PREDICTED_MEANS = [[0.001, 0.002], [0.00113, 0.00026]]
NEXT_PREDICTED_MEAN = [9.34e-4, 4.2e-5]

# When sample is set to one, the predicted means get added to the input noise (which has zero mean)
#    mu0 = [.001, 0.002]
#    mu1 = a*initial_value + b*(obs0+mu0) + c*sample_mean = [.00123, .00066]
#    mu2 = a*mu1 + b*(obs1+mu1) + c*sample_mean = [0.001137, 0.0004540]
SAMPLE_PREDICTED_MEANS = [[0.001, 0.002], [0.00123, 0.00066]]
SAMPLE_NEXT_PREDICTED_MEAN = [0.001137, 0.0004540]

# Here's a set of parameters that are not valid because their dimensions do not conform
ARMA_INVALID_PARAMETERS = {
    "a": [0.8, 0.7, 0.6],
    "b": [0.1, 0.2, 0.3, 0.4],
    "c": [0.03, 0.07, 0.11],
    "d": [2.0, 2.0],
    "sample_mean": [0.001, -0.002, 0.003],
}


def test_ARMA_mean_model():
    default_initial_value = torch.tensor(DEFAULT_INITIAL_VALUE)
    observations = torch.tensor(ARMA_OBSERVATIONS)
    # For now, also use observations as the nois einput when sample=True.
    noise = observations

    # For valid parameters with `default_initial_value` and
    # `observations`, These are the expected outputs below
    predicted_means = torch.tensor(PREDICTED_MEANS)
    next_predicted_mean = torch.tensor(NEXT_PREDICTED_MEAN)
    sample_predicted_means = torch.tensor(SAMPLE_PREDICTED_MEANS)
    sample_next_predicted_mean = torch.tensor(SAMPLE_NEXT_PREDICTED_MEAN)

    mean_model = ARMAMeanModel()

    # Check that certain methods fail before parameters have been initialized

    with pytest.raises(RuntimeError):
        mean_model.get_optimizable_parameters()
    with pytest.raises(RuntimeError):
        mean_model._predict(observations, sample=False)

    utils.set_and_check_parameters(
        mean_model, observations, ARMA_VALID_PARAMETERS, 5, 4
    )

    # CASE 1: Specified parameters, specified input, and specified initial value.

    means, next_mean = mean_model._predict(
        observations, sample=False, mean_initial_value=ARMA_MEAN_INITIAL_VALUE
    )
    assert utils.tensors_about_equal(means, predicted_means)
    assert utils.tensors_about_equal(next_mean, next_predicted_mean)

    # CASE 2: Same but use default initial value.  We only check that
    # the initial value was used.

    means, next_mean = mean_model._predict(observations, sample=False)
    assert torch.all(means[0, :] == default_initial_value)

    # CASE 3: Specified parameters, specified input, specified initial
    # value, and sample == True

    means, next_mean = mean_model._predict(
        observations, sample=True, mean_initial_value=ARMA_MEAN_INITIAL_VALUE
    )
    assert utils.tensors_about_equal(means, sample_predicted_means)
    assert utils.tensors_about_equal(next_mean, sample_next_predicted_mean)

    # Calling sample() on a mean model isn't allowed.
    with pytest.raises(Exception):
        mean_model.sample(noise, initial_mean=None)

    with pytest.raises(ValueError):
        mean_model.set_parameters(**ARMA_INVALID_PARAMETERS)

    print("Done")
