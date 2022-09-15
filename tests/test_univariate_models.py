import pytest

# Standard imports
from math import sqrt

import numpy as np
import torch

# Local modules
# From package under test
from mvarch.mean_models import ARMAMeanModel
from mvarch.univariate_models import (
    marginal_conditional_log_likelihood,
    UnivariateUnitScalingModel,
    UnivariateARCHModel,
)

# Local testing tools
from . import utils

MANUAL_SEED = 42


def check_constant_prediction(
    model, observations, constant_scale_value, constant_mean_value
):
    expanded_constant_scale_value = constant_scale_value.unsqueeze(0).expand(
        observations.shape
    )
    expanded_constant_mean_value = constant_mean_value.unsqueeze(0).expand(
        observations.shape
    )

    next_scale, _, scale, __ = model.predict(observations)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    next_scale, scale = model._predict(observations, sample=False)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    next_scale, scale = model._predict(observations, sample=True)
    assert torch.all(scale == expanded_constant_scale_value)
    assert torch.all(next_scale == constant_scale_value)

    # Confirm that sample returns the original observations plus the mean model output
    output = model.sample(observations)[0]
    assert utils.tensors_about_equal(
        output, observations + expanded_constant_mean_value
    )


@pytest.fixture
def univariate_unit_scaling_model():
    return UnivariateUnitScalingModel()


def test_unitialized_unit_scaling_model_sample_raises(univariate_unit_scaling_model):
    with pytest.raises(ValueError):
        univariate_unit_scaling_model.sample(10)


def test_unit_scaling_model_fit_raises(univariate_unit_scaling_model):
    N = 3
    observations = torch.randn((10, N), dtype=torch.float)
    with pytest.raises(ValueError):
        univariate_unit_scaling_model.fit(observations)


def test_unit_scaling_model_operates_sanely(univariate_unit_scaling_model):
    N = 3
    observations = torch.randn((10, N), dtype=torch.float)
    noise = torch.randn(observations.shape, dtype=torch.float)

    assert univariate_unit_scaling_model.is_optimizable == False

    utils.set_and_check_parameters(
        univariate_unit_scaling_model, observations, {"dim": N}, 1, 0
    )

    check_constant_prediction(
        univariate_unit_scaling_model,
        observations,
        torch.ones(observations.shape[1], dtype=torch.float),
        torch.zeros(observations.shape[1], dtype=torch.float),
    )


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


@pytest.fixture
def univariate_arch_model():
    return UnivariateARCHModel()


def test_uninitialized_arch_model_get_optimizable_raises(univariate_arch_model):
    with pytest.raises(RuntimeError):
        univariate_arch_model.get_optimizable_parameters()


def test_uninitialized_arch_model_sample_raises(univariate_arch_model):
    observations = torch.tensor(ARCH_CENTERED_OBSERVATIONS, dtype=torch.float)
    with pytest.raises(RuntimeError):
        univariate_arch_model.sample(observations)


def test_arch_model(univariate_arch_model):
    default_initial_value = torch.tensor(ARCH_DEFAULT_INITIAL_VALUE, dtype=torch.float)
    arch_scale_initial_value = torch.tensor(ARCH_SCALE_INITIAL_VALUE, dtype=torch.float)
    observations = torch.tensor(ARCH_CENTERED_OBSERVATIONS, dtype=torch.float)
    # For now, also use observations as the nois einput when sample=True.
    noise = observations

    assert univariate_arch_model.is_optimizable == True

    utils.set_and_check_parameters(
        univariate_arch_model, observations, ARCH_VALID_PARAMETERS, 5, 4
    )

    # Case 1: _predict with sample=False and specified initial value
    scale_next, scale = univariate_arch_model._predict(
        observations, scale_initial_value=arch_scale_initial_value
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
    sample_scale_next, sample_scale = univariate_arch_model._predict(
        observations, sample=True, scale_initial_value=arch_scale_initial_value
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
    scale_next, scale = univariate_arch_model._predict(observations)
    assert utils.tensors_about_equal(
        scale[0, :], torch.tensor(ARCH_DEFAULT_INITIAL_VALUE)
    )

    with pytest.raises(ValueError):
        univariate_arch_model.set_parameters(**ARCH_INVALID_PARAMETERS)


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
    CONSTANT_SCALE = torch.tensor(0.25)
    CONSTANT_MEAN = torch.tensor(0.5)
    SAMPLE_SIZE = 2500
    TOLERANCE = 0.075

    # The tolerance hasn't been chosen very scientifically.  The sample size
    # Is fairly small for a quick test so it won't be tight.

    torch.manual_seed(MANUAL_SEED)
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

    scale_next, mean_next = model.predict(random_observations)[:2]

    print("mean prediction: ", mean_next)
    print("scale prediction: ", scale_next)

    assert utils.tensors_about_equal(scale_next, CONSTANT_SCALE, TOLERANCE)
    assert utils.tensors_about_equal(mean_next, CONSTANT_MEAN, TOLERANCE)

    # Make sure that sample(int) returns something reasonable
    sample_output = model.sample(SAMPLE_SIZE)[0]
    sample_mean = float(torch.mean(sample_output))
    sample_std = float(torch.std(sample_output))

    print("sample mean: ", sample_mean)
    print("sample std: ", sample_std)

    assert utils.tensors_about_equal(sample_std, CONSTANT_SCALE, TOLERANCE)
    assert utils.tensors_about_equal(sample_mean, CONSTANT_MEAN, TOLERANCE)

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


SANE_UV_PARAMETERS = {
    "a": [0.9482781, 0.87155545],
    "b": [0.15232861, 0.10627481],
    "c": [0.2793683, 0.48152938],
    "d": [0.01429591, 0.71451986],
    "sample_scale": [0.2501, 0.1000],
}


SANE_MEAN_PARAMETERS = {
    "a": [0.16967115, 0.1777586],
    "b": [0.01160183, 0.02662377],
    "c": [0.81673557, 0.796202],
    "d": [0.83479756, 0.59621435],
    "sample_mean": [0.5, -0.24999999],
}


@pytest.fixture()
def sane_model():
    mean_model = ARMAMeanModel()
    mean_model.set_parameters(**SANE_MEAN_PARAMETERS)

    univariate_model = UnivariateARCHModel(mean_model=mean_model)
    univariate_model.set_parameters(**SANE_UV_PARAMETERS)

    return univariate_model


def generate_observations(
    sample_size,
    mean_vector: torch.tensor,
    uv_scale: torch.tensor,
) -> torch.tensor:
    """
    Generate independent random observations
    """
    if mean_vector.shape != uv_scale.shape:
        raise ValueError("Vectors must conform")

    torch.manual_seed(MANUAL_SEED)
    white_noise = torch.randn((sample_size, mean_vector.shape[0]))
    # Correct sample mean, sample variance, and orthogonality slightly so that the
    # sample statistics are what we want.
    white_noise = white_noise - torch.mean(white_noise, dim=0)
    white_noise = torch.linalg.qr(white_noise, mode="reduced")[0] * sqrt(sample_size)

    # Resulting sample should have 1) zero mean; 2) unit variance; 3) orthogonal columns

    # Multiply and drop the last dimension.
    random_observations = uv_scale * white_noise + mean_vector

    return random_observations, white_noise


@pytest.fixture()
def random_observations():
    SAMPLE_SIZE = 2000  # Was 2500
    CONSTANT_MEAN = torch.tensor([0.5, -0.25])
    CONSTANT_UV_SCALE = torch.tensor([0.25, 0.1])
    random_observations = generate_observations(
        SAMPLE_SIZE,
        CONSTANT_MEAN,
        CONSTANT_UV_SCALE,
    )[0]
    return random_observations


def test_simulate(sane_model, random_observations):
    # Check that simulate executes and returns tensors with the proper dimensions
    output, scale, mean = sane_model.simulate(random_observations, 10, 3)
    assert output.shape == (3, 10, 2)
    assert scale.shape == output.shape
    assert mean.shape == output.shape

    # If simulation count isn't provided, first dimension should be dropped.
    output, scale, mean = sane_model.simulate(random_observations, 10)
    assert output.shape == (10, 2)
    assert scale.shape == output.shape
    assert mean.shape == output.shape
