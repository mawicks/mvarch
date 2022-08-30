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
    scale, scale_next = model._predict(
        observations, scale_initial_value=MVARCH_SCALE_INITIAL_VALUE
    )[:2]

    # ASSERT SOMETHING LIKE THIS

    # assert utils.tensors_about_equal(
    # scale_next, torch.tensor(PREDICTED_SCALE_NEXT, dtype=torch.float)
    #    )

    print("_predict() with sample=False")
    print("scale: ", scale)
    print("scale**2: ", scale**2)
    print("scale_next: ", scale_next)
    print("scale_next**2: ", scale_next**2)

    # Case 2: _predict with sample=True and specified initial value
    sample_scale, sample_scale_next = model._predict(
        observations, sample=True, scale_initial_value=MVARCH_SCALE_INITIAL_VALUE
    )[:2]

    # ASSERT SOMETHING

    print("_predict() with sample=True")
    print("sample_scale: ", sample_scale)
    print("sample_scale**2: ", sample_scale**2)
    print("sample_scale_next: ", sample_scale_next)
    print("sample_scale_next**2: ", sample_scale_next**2)

    # Case 3: _predict with sample=False and using default initial value.
    scale, scale_next = model._predict(observations)

    # ASSERT SOMETHING LIKE THIS
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


def test_arch_fit():
    """As a basic sanity check, test fit a model on white noise with
    constant variance and see if the model predicts that constant
    variance.
    """
    CONSTANT_SCALE = 0.25
    CONSTANT_MEAN = 0.0
    SAMPLE_SIZE = 25  # Was 2500
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

    for univariate_model in (UnivariateARCHModel(), UnivariateUnitScalingModel()):
        model = MultivariateARCHModel(
            constraint=ParameterConstraint.FULL, univariate_model=univariate_model
        )
        model.fit(random_observations)

        # print("mean model parameters: ", model.mean_model.get_parameters())
        print("MV scale model parameters: ", model.get_parameters())

        mv_scale_next = model.predict(random_observations)[3]

        print("MV scale prediction: ", mv_scale_next)

        # assert abs(scale_next - CONSTANT_SCALE) < TOLERANCE * CONSTANT_SCALE
        # assert abs(mean_next - CONSTANT_MEAN) < TOLERANCE * abs(CONSTANT_MEAN)

        # Make sure that sample(int) returns something reasonable
        sample_output = model.sample(SAMPLE_SIZE)[0]
        sample_mean = float(torch.mean(sample_output))
        sample_std = float(torch.std(sample_output))

        print("sample mean: ", sample_mean)
        print("sample std: ", sample_std)

        # assert abs(sample_mean - CONSTANT_MEAN) < TOLERANCE * CONSTANT_MEAN
        # assert abs(sample_std - CONSTANT_SCALE) < TOLERANCE * CONSTANT_SCALE

        # Check that the log likelihoods being returned are reasonable
        actual = model.mean_log_likelihood(random_observations)

        # What should this be?  -0.5 E[x**2] / (sigma**2) - 0.5*log(2*pi) - log(sigma)
        # Which is -.5(1+log(2*pi)) - log(sigma)

        expected = -0.5 * (1 + np.log(2 * np.pi)) - np.log(CONSTANT_SCALE)

        print("actual: ", actual)
        print("expected: ", expected)

        # Since these are logs, we use an absolute tolerance rather than a relative tolerance
        # Again, this is just a sanity check and not a very stringent test.
        # assert abs(actual - expected) < TOLERANCE
