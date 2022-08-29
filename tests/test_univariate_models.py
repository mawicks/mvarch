import pytest

# Standard imports
from math import sqrt

import numpy as np
import torch

# Local modules
from mvarch.univariate_models import (
    marginal_conditional_log_likelihood,
    UnivariateUnitScalingModel,
    UnivariateARCHModel,
)

from mvarch.mean_models import ARMAMeanModel

from . import utils


def check_constant_prediction(
    model, observations, constant_scale_value, constant_mean_value
):
    expanded_constant_scale_value = constant_scale_value.unsqueeze(0).expand(
        observations.shape
    )
    expanded_constant_mean_value = constant_mean_value.unsqueeze(0).expand(
        observations.shape
    )

    scale, _, next_scale, __ = model.predict(observations)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    scale, next_scale = model._predict(observations, sample=False)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    scale, next_scale = model._predict(observations, sample=True)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    # Confirm that sample returns the original observations plus the mean model output
    output = model.sample(observations)[0]
    assert utils.tensors_about_equal(
        output, observations + expanded_constant_mean_value
    )


def test_scaling_model():
    N = 3
    observations = torch.randn((10, N), dtype=torch.float)
    noise = torch.randn(observations.shape, dtype=torch.float)
    model = UnivariateUnitScalingModel()

    with pytest.raises(ValueError):
        model.sample(10)

    utils.set_and_check_parameters(model, observations, {"n": N}, 0, 0)

    check_constant_prediction(
        model,
        observations,
        torch.ones(observations.shape[1], dtype=torch.float),
        torch.zeros(observations.shape[1], dtype=torch.float),
    )


def test_constant_scale_with_armma_mean_fails():
    mean_model = ARMAMeanModel()
    with pytest.raises(ValueError):
        model = UnivariateUnitScalingModel(mean_model=mean_model)


# These aren't realistic values.  They're just some nice whole numbers for testing.
# We use a dimension of two with independent values because otherwise we wouldn't catch
# transpose errors or stacking errors for example.
ARCH_VALID_PARAMETERS = {
    "a": [2, 3],
    "b": [5, 7],
    "c": [11, 13],
    "d": [17, 19],
    "sample_scale": [23, 29],
}
CONSTANT_MEAN = 7
ARCH_SCALE_INITIAL_VALUE = [31, 37]
ARCH_CENTERED_OBSERVATIONS = [[39, 41], [0, 0]]
ARCH_DEFAULT_INITIAL_VALUE = [391, 551]  # This is d * sample_scale

# Here's what should happen:
#    sigma0 = [31, 37] # Supplied initial value
#    [a*sigma0, b*obs0,c*sample_scale]:
#        [[62, 111], [195, 287], [253, 377]] -> sqrt([105878, 236819])
#    This leads to sigma1 = sqrt([105878, 236819]) = [325.38..., 486.64...]
#    Next iteration:
#    [a*sigma1, b*obs1, c*sample_scale]:
#       [[2*sqrt(105878), 3*sqrt(236819)], [0, 0], [253, 377]
#    This leads to sigma2 = sqrt([487521, 2273500]) = [698.22..., 1507.81...]

PREDICTED_SCALE = [[31, 37], [sqrt(105878), sqrt(236819)]]
PREDICTED_SCALE_NEXT = [sqrt(487521), sqrt(2273500)]

# When sample is set to one, the input is scaled by the predicted scale.
#    sigma0 = [31, 37] # Supplied initial value
#    [a*sigma0, b*(sigma0*obs0), c*sample_scale]:
#        [[62, 111], [6045, 10619], [253, 377]] -> sqrt([36609878, 112917611])
#    Next iteration:
#         [[2*sqrt(36609878), 3*sqrt(112917611)], [0, 0], [253, 377]] -> sqrt([[146503521, 1016400628])

SAMPLE_PREDICTED_SCALE = [[31, 37], [sqrt(36609878), sqrt(112917611)]]
SAMPLE_PREDICTED_SCALE_NEXT = [sqrt(146503521), sqrt(1016400628)]

# Here's a set of parameters that are not valid because their dimensions do not conform
ARCH_INVALID_PARAMETERS = {
    "a": [0.8, 0.7, 0.6],
    "b": [0.1, 0.2, 0.3, 0.4],
    "c": [0.03, 0.07, 0.11],
    "d": [2.0, 2.0],
    "sample_scale": [0.001, 0.002, 0.003],
}


def test_arch_model():
    default_initial_value = torch.tensor(ARCH_DEFAULT_INITIAL_VALUE, dtype=torch.float)
    observations = torch.tensor(ARCH_CENTERED_OBSERVATIONS, dtype=torch.float)
    # For now, also use observations as the nois einput when sample=True.
    noise = observations

    model = UnivariateARCHModel()
    with pytest.raises(ValueError):
        model.get_optimizable_parameters()

    with pytest.raises(RuntimeError):
        model.sample(observations)

    utils.set_and_check_parameters(model, observations, ARCH_VALID_PARAMETERS, 5, 4)

    # Case 1: _predict with sample=False and specified initial value
    scale, scale_next = model._predict(
        observations, scale_initial_value=ARCH_SCALE_INITIAL_VALUE
    )
    assert utils.tensors_about_equal(
        scale, torch.tensor(PREDICTED_SCALE, dtype=torch.float)
    )
    assert utils.tensors_about_equal(
        scale_next, torch.tensor(PREDICTED_SCALE_NEXT, dtype=torch.float)
    )
    print("_predict() with sample=False")
    print("scale: ", scale)
    print("scale**2: ", scale**2)
    print("scale_next: ", scale_next)
    print("scale_next**2: ", scale_next**2)

    # Case 2: _predict with sample=True and specified initial value
    sample_scale, sample_scale_next = model._predict(
        observations, sample=True, scale_initial_value=ARCH_SCALE_INITIAL_VALUE
    )
    assert utils.tensors_about_equal(
        sample_scale, torch.tensor(SAMPLE_PREDICTED_SCALE, dtype=torch.float)
    )
    assert utils.tensors_about_equal(
        sample_scale_next, torch.tensor(SAMPLE_PREDICTED_SCALE_NEXT, dtype=torch.float)
    )
    print("_predict() with sample=True")
    print("sample_scale: ", sample_scale)
    print("sample_scale**2: ", sample_scale**2)
    print("sample_scale_next: ", sample_scale_next)
    print("sample_scale_next**2: ", sample_scale_next**2)

    # Case 3: _predict with sample=False and using default initial value.
    scale, scale_next = model._predict(observations)
    assert utils.tensors_about_equal(
        scale[0, :], torch.tensor(ARCH_DEFAULT_INITIAL_VALUE)
    )

    with pytest.raises(ValueError):
        model.set_parameters(**ARCH_INVALID_PARAMETERS)


def test_marginal_likelihood():
    """Test marginal_conditional_log_likelihood, which computes the log
    probability for a number of different scales all at once.  Do the
    equivalent calculation by looping over the scale and values and
    constructing a distribution object for each specific scale parameter
    before calling log_prob.  This should yield the same result
    as marginal_conditional_log_likelhood()

    """
    # We'll use the normal distribution, but marginal_condition_log_likelihood
    # should work for any distribution.
    dist = torch.distributions.Normal

    # General random scales, then generate standard normal variables and scale them.
    scale = torch.abs(torch.randn(10, 3))
    noise = scale * torch.randn(10, 3)

    # Compute the log likelihood for the scaled variables using the
    # correct scale in the distribution.  We sum likelihoods over
    # columns and average over rows.
    ll = []
    for noise_row, scale_row in zip(noise, scale):
        ll.append(
            sum(
                [
                    dist(loc=0.0, scale=s).log_prob(n)
                    for n, s in zip(noise_row, scale_row)
                ]
            )
        )

    expected = float(np.mean(ll))

    actual = float(
        marginal_conditional_log_likelihood(
            noise, scale, distribution=dist(loc=0.0, scale=1.0)
        )
    )

    assert abs(expected - actual) < utils.EPS * abs(expected)


def test_arch_fit():
    """As a basic sanity check, test fit a model on white noise with
    constant variance and see if the model predicts that constant
    variance.
    """
    CONSTANT_SCALE = 0.25
    CONSTANT_MEAN = 0.5
    SAMPLE_SIZE = 2500
    TOLERANCE = 0.075

    # The tolerance hasn't been chosen very scientifically.  The sample size
    # is fairly small for a quick test so it won't be tight.

    random_observations = torch.randn((SAMPLE_SIZE, 1))

    # The sample variance is random.  Scale the sample so that the
    # sample standard deviation and the sample mean are *exactly* what
    # we expect the model to predict.
    random_observations = (
        random_observations - torch.mean(random_observations)
    ) / torch.std(random_observations) * CONSTANT_SCALE + CONSTANT_MEAN

    model = UnivariateARCHModel(mean_model=ARMAMeanModel())
    model.fit(random_observations)

    print("mean model parameters: ", model.mean_model.get_parameters())
    print("scale model parameters: ", model.get_parameters())

    scale_next, mean_next = model.predict(random_observations)[2:]

    print("mean prediction: ", mean_next)
    print("scale prediction: ", scale_next)

    assert abs(scale_next - CONSTANT_SCALE) < TOLERANCE * CONSTANT_SCALE
    assert abs(mean_next - CONSTANT_MEAN) < TOLERANCE * abs(CONSTANT_MEAN)

    # Make sure that sample(int) returns something reasonable
    sample_std = float(torch.std(model.sample(SAMPLE_SIZE)[0]))
    print("std sample: ", sample_std)
    assert abs(sample_std - CONSTANT_SCALE) < TOLERANCE * CONSTANT_SCALE

    # Check that the log likelihoods being returned are reasonable
    actual = model.mean_log_likelihood(random_observations)

    # What should this be?  -0.5 E[x**2] / (sigma**2) - 0.5*log(2*pi) - log(sigma)
    # Which is -.5(1+log(2*pi)) - log(sigma)

    expected = -0.5 * (1 + np.log(2 * np.pi)) - np.log(CONSTANT_SCALE)

    print("actual: ", actual)
    print("expected: ", expected)

    # Since these are logs, we use an absolute tolerance rather than a relative tolerance
    # Again, this is just a sanity check and not a very stringent test.
    assert abs(actual - expected) < TOLERANCE
