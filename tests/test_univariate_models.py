import pytest

import torch

# Local modules
from mvarch.univariate_models import UnivariateUnitScalingModel, UnivariateARCHModel
from . import utils


def test_scaling_model():
    N = 3
    observations = torch.randn((10, N))
    noise = torch.randn(observations.shape)
    model = UnivariateUnitScalingModel()

    utils.set_and_check_parameters(model, observations, {"n": N}, 0, 0)


ARCH_GOOD_PARAMETERS = {
    "a": [0.8, 0.7],
    "b": [0.1, 0.2],
    "c": [0.03, 0.07],
    "d": [2.0, 2.0],
    "sample_scale": [0.001, -0.002],
}
ARCH_SCALE_INITIAL_VALUE = [0.001, 0.002]
DEFAULT_INITIAL_VALUE = [0.002, -0.004]  # This is d * sample_mean
ARCH_OBSERVATIONS = [
    [0.003, -0.005],
    [0.0, 0.0],
]


def test_arch_odel():
    default_initial_value = torch.tensor(DEFAULT_INITIAL_VALUE)
    observations = torch.tensor(ARCH_OBSERVATIONS)
    # For now, also use observations as the nois einput when sample=True.
    noise = observations

    model = UnivariateARCHModel()
    utils.set_and_check_parameters(model, observations, ARCH_GOOD_PARAMETERS, 5, 4)


# Here's what should happen:
#    mu0 = [.001, 0.002]
#    mu1 = a*initial_value + b*obs0 + c*sample_mean = [.00113, .0003]
#    mu2 = a*mu1 + b*obs1 + c*sample_mean = [9.34e-4, 4.2e-5]
PREDICTED_SCALE = [[0.001, 0.002], [0.00113, 0.00026]]
NEXT_PREDICTED_SCALE = [9.34e-4, 4.2e-5]

# When sample is set to one, the predicted means get added to the input noise (which has zero mean)
#    mu0 = [.001, 0.002]
#    mu1 = a*initial_value + b*(obs0+mu0) + c*sample_mean = [.00123, .00066]
#    mu2 = a*mu1 + b*(obs1+mu1) + c*sample_mean = [0.001137, 0.0004540]
SAMPLE_PREDICTED_SCALE = [[0.001, 0.002], [0.00123, 0.00066]]
SAMPLE_NEXT_PREDICTED_SCALE = [0.001137, 0.0004540]

# Here's a set of parameters that are not valid because their dimensions do not conform
ARCH_INVALID_PARAMETERS = {
    "a": [0.8, 0.7, 0.6],
    "b": [0.1, 0.2, 0.3, 0.4],
    "c": [0.03, 0.07, 0.11],
    "d": [2.0, 2.0],
    "sample_mean": [0.001, -0.002, 0.003],
}
