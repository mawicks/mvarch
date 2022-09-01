import pytest

# Standard imports
from math import sqrt

import numpy as np
import torch

# Local modules
# From package under test
from mvarch.mean_models import ARMAMeanModel, ZeroMeanModel
from mvarch.univariate_models import (
    marginal_conditional_log_likelihood,
    UnivariateUnitScalingModel,
    UnivariateARCHModel,
)
from mvarch.multivariate_models import (
    joint_conditional_log_likelihood,
    MultivariateARCHModel,
)
from mvarch.parameters import ParameterConstraint

# Local testing tools.
from . import utils


# These aren't realistic values.  They're just some nice whole numbers for testing.
# We use a dimension of two with independent values because otherwise we wouldn't catch
# transpose errors or stacking errors for example.
MVARCH_VALID_PARAMETERS = {
    "a": [2, 3],
    "b": [5, 7],
    "c": [11, 13],
    "d": [17, 19],
    "sample_scale": [[23, 0], [29, 31]],
}
CONSTANT_MEAN = 7
MVARCH_SCALE_INITIAL_VALUE = [[31, 0], [37, 39]]
MVARCH_INVALID_SCALE_INITIAL_VALUE = [[31, 0]]
MVARCH_CENTERED_OBSERVATIONS = [[39, 41], [0, 0]]
MVARCH_DEFAULT_INITIAL_VALUE = [[391, 0], [551, 589]]  # This is d * sample_scale

# Here's what should happen:
#    sigma0 = [[31, 0], [37, 39]] # Supplied initial value
#    [a*sigma0;  b*obs0;  c*sample_scale]:
#        [[62, 0], [111, 117]];   [195, 287];  [[253, 0], [377, 403]]
#    i.e., [[ 62,    0, 195, 253,   0]
#           [ 111, 117, 287, 377, 403]]
#    so that h @ h.T = [[ 105878, 158228], [158228, 412917]]
#    Next iteration:
#    [a*sigma1, b*obs1, c*sample_scale]:

PREDICTED_SCALE_SQUARED = [[105878, 158228], [158228, 412917]]

# When sample is set to one, the input is scaled by the predicted scale.
#    scale0 = [[31, 0], [37, 39]] # Supplied initial value
#    [a*scale0, b*(scale0*obs0), c*sample_scale]:
#        [[62, 0], [111, 117]];  [6045, 21294];  [[253, 0], [377, 402]]
#    i.e., [[ 62,    0, 6045, 253,   0]
#           [ 111, 117, 21294, 377, 403]]
#    so that h @ h.T = [[ 36609878, 128824493], [128824493, 453772904]]

SAMPLE_PREDICTED_SCALE_SQUARED = [[36609878, 128824493], [128824493, 453772904]]

# Here's a set of parameters that are not valid because their dimensions do not conform
MVARCH_INVALID_PARAMETERS1 = {
    "a": [0.8, 0.7, 0.6],
    "b": [0.1, 0.2, 0.3, 0.4],
    "c": [0.03, 0.07, 0.11],
    "d": [2.0, 2.0],
    "sample_scale": [0.001, 0.002, 0.003],
}
MVARCH_INVALID_PARAMETERS2 = {
    "a": [0.8, 0.7],
    "b": [0.1, 0.2],
    "c": [0.03, 0.07],
    "d": [2.0, 2.0],
    "sample_scale": [0.001, 0.002],
}
MVARCH_INVALID_PARAMETERS = [MVARCH_INVALID_PARAMETERS1, MVARCH_INVALID_PARAMETERS2]


def test_mvarch_model():
    default_initial_value = torch.tensor(
        MVARCH_DEFAULT_INITIAL_VALUE, dtype=torch.float
    )
    observations = torch.tensor(MVARCH_CENTERED_OBSERVATIONS, dtype=torch.float)
    # For now, also use observations as the nois einput when sample=True.
    noise = observations

    # Create a MV model with diagonal parmaeters.
    # For coverage, make sure we can construct a model with each possible parameter type:

    for constraint in ParameterConstraint.__members__.values():
        model = MultivariateARCHModel(constraint=constraint)

    model = MultivariateARCHModel(constraint=ParameterConstraint.DIAGONAL)

    with pytest.raises(RuntimeError):
        model.get_optimizable_parameters()

    with pytest.raises(RuntimeError):
        model.sample(observations)

    utils.set_and_check_parameters(model, observations, MVARCH_VALID_PARAMETERS, 5, 4)

    # Case 1: _predict with sample=False and specified initial value
    scale = model._predict(
        observations, scale_initial_value=MVARCH_SCALE_INITIAL_VALUE
    )[0]

    print("_predict() with sample=False")

    assert utils.tensors_about_equal(
        scale[0], torch.tensor(MVARCH_SCALE_INITIAL_VALUE, dtype=torch.float)
    )

    assert utils.tensors_about_equal(
        scale[1] @ scale[1].T, torch.tensor(PREDICTED_SCALE_SQUARED, dtype=torch.float)
    )

    # Case 2: _predict with sample=True and specified initial value
    sample_scale = model._predict(
        observations, sample=True, scale_initial_value=MVARCH_SCALE_INITIAL_VALUE
    )[0]

    print("_predict() with sample=True")

    assert utils.tensors_about_equal(
        sample_scale[0], torch.tensor(MVARCH_SCALE_INITIAL_VALUE, dtype=torch.float)
    )

    sample_scale_squared = sample_scale[1] @ sample_scale[1].T
    assert utils.tensors_about_equal(
        sample_scale_squared,
        torch.tensor(SAMPLE_PREDICTED_SCALE_SQUARED, dtype=torch.float),
        eps=1e-5,  # Use a slightly larger eps for this test because of the matrix "squaring"
    )

    # Case 3: _predict with sample=False and using default initial value.
    scale = model._predict(observations)[0]

    assert utils.tensors_about_equal(
        scale[0, :], torch.tensor(MVARCH_DEFAULT_INITIAL_VALUE)
    )

    # For coverage, try several bad parameter values with gates:
    for p in MVARCH_INVALID_PARAMETERS:
        with pytest.raises(ValueError):
            model.set_parameters(**p)

    with pytest.raises(ValueError):
        model._predict(
            observations, scale_initial_value=MVARCH_INVALID_SCALE_INITIAL_VALUE
        )


def test_joint_log_likelihood():
    """Test joint_conditional_log_likelihood, which computes the log
    probability for a number of different scales and correlations all
    at once.  Do the equivalent calculation by looping over the scale
    and values and constructing a distribution object for each
    specific scale parameter before calling log_prob.  This should
    yield the same result as marginal_conditional_log_likelhood()

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

    # Do something like this
    # actual = float(
    #    marginal_conditional_log_likelihood(
    #        noise, scale, distribution=dist(loc=0.0, scale=1.0)
    # )
    # )

    # ASSERT SOMETHING LIKE THIS

    # assert abs(expected - actual) < utils.EPS * abs(expected)


def generate_observations(
    sample_size,
    mean_vector: torch.tensor,
    uv_scale: torch.tensor,
    mv_scale: torch.tensor,
) -> torch.tensor:
    """
    Generate two sequences with a correlation of `correlation`
    """
    if (
        mean_vector.shape != uv_scale.shape
        or mean_vector.shape[0] != mv_scale.shape[0]
        or mv_scale.shape[0] != mv_scale.shape[1]
    ):
        raise ValueError("Vectors must conform")

    white_noise = torch.randn((sample_size, mean_vector.shape[0]))
    # Correct sample mean, sample variance, and orthogonality slightly so that the
    # sample statistics are what we want.
    white_noise = white_noise - torch.mean(white_noise, dim=0)
    white_noise = torch.linalg.qr(white_noise, mode="reduced")[0] * sqrt(sample_size)

    # Resulting sample should have 1) zero mean; 2) unit variance; 3) orthogonal columns

    # Expand everyhing to be either (sample_size, n, 1) or (sample_size, n, n) to make
    # dimensions conform and to remove any ambiguity in the matrix multiply.

    white_noise = white_noise.unsqueeze(2).expand(
        (sample_size, mean_vector.shape[0], 1)
    )
    mean_vector = mean_vector.unsqueeze(0).unsqueeze(2).expand(white_noise.shape)
    uv_scale = uv_scale.unsqueeze(0).unsqueeze(2).expand(white_noise.shape)
    # Construct sample matrix and expand scale_matrix along sample dimension
    scale_matrix = mv_scale.unsqueeze(0).expand((sample_size,) + mv_scale.shape)

    # Multiply and drop the last dimension.
    random_observations = (
        uv_scale * (scale_matrix @ white_noise) + mean_vector
    ).squeeze(2)
    return random_observations


def observation_stats(observations):
    mean = torch.mean(observations, dim=0)
    centered_observations = observations - torch.mean(observations, dim=0)
    print("mean of observations: ", mean)
    print("std of observations: ", torch.std(centered_observations, dim=0))
    cov = (
        centered_observations.T @ centered_observations
    ) / centered_observations.shape[0]
    print("cov of observations: ", cov)


def make_model(univariate_model_type):
    MEAN_MODEL_TYPE = {
        UnivariateARCHModel: ARMAMeanModel,
        UnivariateUnitScalingModel: ZeroMeanModel,
    }

    univariate_model = univariate_model_type(
        mean_model=MEAN_MODEL_TYPE[univariate_model_type]()
    )

    model = MultivariateARCHModel(
        constraint=ParameterConstraint.FULL, univariate_model=univariate_model
    )
    return model


def test_arch_fit():
    """As a basic sanity check, test fit a model on white noise with
    constant variance and see if the model predicts that constant
    variance.
    """
    CONSTANT_MEAN = {
        UnivariateARCHModel: torch.tensor([0.5, -0.25]),
        UnivariateUnitScalingModel: torch.tensor([0.0, 0.0]),
    }

    CONSTANT_UV_SCALE = torch.tensor([0.25, 0.1])
    SAMPLE_SIZE = 1000  # Was 2500
    CORRELATION = -0.5
    CONSTANT_MV_SCALE = torch.tensor(
        [[1.0, 0.0], [CORRELATION, sqrt(1.0 - CORRELATION**2)]]
    )
    TOLERANCE = 0.075

    univariate_model_type = UnivariateARCHModel
    random_observations = generate_observations(
        SAMPLE_SIZE,
        CONSTANT_MEAN[univariate_model_type],
        CONSTANT_UV_SCALE,
        CONSTANT_MV_SCALE,
    )

    observation_stats(random_observations)
    model = make_model(univariate_model_type)
    model.fit(random_observations)

    # print("mean model parameters: ", model.mean_model.get_parameters())
    print("MV scale model parameters: ", model.get_parameters())

    mv_scale_next, uv_scale_next, uv_mean_next = model.predict(random_observations)[3:]

    print("UV mean prediction: ", uv_mean_next)
    print("UV scale prediction: ", uv_scale_next)
    print("MV scale prediction: ", mv_scale_next)

    assert utils.tensors_about_equal(
        uv_mean_next, CONSTANT_MEAN[univariate_model_type], TOLERANCE
    )
    assert utils.tensors_about_equal(uv_scale_next, CONSTANT_UV_SCALE, TOLERANCE)
    assert utils.tensors_about_equal(mv_scale_next, CONSTANT_MV_SCALE, TOLERANCE)

    # Make sure that sample(int) returns something reasonable
    sample_output = model.sample(SAMPLE_SIZE)[0]
    sample_mean = torch.mean(sample_output, dim=0)
    sample_std = torch.std(sample_output, dim=0)
    sample_corrcoef = torch.corrcoef(sample_output.T)

    print("sample mean: ", sample_mean)
    print("sample std: ", sample_std)
    print("sample corrcoef: ", sample_corrcoef)

    assert utils.tensors_about_equal(
        sample_mean, CONSTANT_MEAN[univariate_model_type], TOLERANCE
    )
    assert utils.tensors_about_equal(sample_std, CONSTANT_UV_SCALE, TOLERANCE)
    assert utils.tensors_about_equal(
        sample_corrcoef, CONSTANT_MV_SCALE @ CONSTANT_MV_SCALE.T, TOLERANCE
    )

    # Check that the log likelihoods being returned are reasonable
    actual = model.mean_log_likelihood(random_observations)
    print("actual: ", actual)

    # FIXME
    # What should this be?  -0.5 E[x**2] / (sigma**2) - 0.5*log(2*pi) - log(sigma)
    # Which is -.5(1+log(2*pi)) - log(sigma)
    # expected = torch.sum(-0.5 * (1 + np.log(2 * np.pi)) - torch.log(CONSTANT_UV_SCALE))
    # print("expected: ", expected)

    # Since these are logs, we use an absolute tolerance rather than a relative tolerance
    # Again, this is just a sanity check and not a very stringent test.
    # assert abs(actual - expected) < TOLERANCE

    # This is here for coverage.  We're not checking the return values.
    model = make_model(UnivariateUnitScalingModel)
    model.fit(random_observations)
