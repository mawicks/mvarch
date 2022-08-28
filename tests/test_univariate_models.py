import pytest

import torch

# Local modules
from mvarch.univariate_models import UnivariateUnitScalingModel, UnivariateARCHModel
from . import utils


def test_scaling_model():
    observations = torch.randn((10, 3))
    noise = torch.randn(observations.shape)
    model = UnivariateUnitScalingModel()


def test_arch_odel():
    observations = torch.randn((10, 3))
    noise = torch.randn(observations.shape)
    model = UnivariateARCHModel()


ARMA_GOOD_PARAMETERS = {
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
