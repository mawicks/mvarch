# Standard Python
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, Union

# Common packages
import torch

# Local modules
from . import constants
from .distributions import Distribution, NormalDistribution, StudentTDistribution
from .parameters import (
    ParameterConstraint,
    Parameter,
    ScalarParameter,
    DiagonalParameter,
    TriangularParameter,
    FullParameter,
)

from .univariate_models import (
    UnivariateScalingModel,
    UnivariateUnitScalingModel,
    UnivariateARCHModel,
)
from .matrix_ops import make_diagonal_nonnegative
from .optimize import optimize


def joint_conditional_log_likelihood(
    observations: torch.Tensor,
    mv_scale: torch.Tensor,
    distribution: torch.distributions.Distribution,
    uv_scale=Union[torch.Tensor, None],
) -> torch.Tensor:
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       mv_scale: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.Distribution or other object with a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.
       uv_scale: torch.Tensor of shape (n_obj, n_symbols) or None.
              When uv_scale is specified, the observed variables
              are related to the innovations by x = diag(sigma) T e.
              This is when the estimator for the transformation
              from e to x is factored into a diagonal scaling
              matrix (sigma) and a correlating transformation T.
              When sigma is not specified any scaling is already embedded in T.


    Returns:
       torch.Tensor - the mean (per sample) log_likelihood"""

    # Divide by the determinant by subtracting its log to get the the log
    # likelihood of the observations.  The `transformations` are lower-triangular, so
    # only the diagonal entries need to be used in the determinant calculation.
    # This should be faster than calling log_det().

    log_det = torch.log(torch.abs(torch.diagonal(mv_scale, dim1=1, dim2=2)))

    if uv_scale is not None:
        log_det = log_det + torch.log(torch.abs(uv_scale))
        observations = observations / uv_scale

    # First get the innovations sequence by forming transformation^(-1)*observations
    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `t` is (n_obs, n, n) while
    # the shape of `observations` is (n_obs, n). Without adding the
    # extract dimension to observations, the solver won't see conforming dimensions.
    # We remove the ambiguity, by making observations` have shape (n_obj, n, 1), then
    # we remove the extra dimension from e.
    e = torch.linalg.solve_triangular(
        mv_scale,
        observations.unsqueeze(2),
        upper=False,
    ).squeeze(2)

    logging.debug(f"e: \n{e}")

    # Compute the log likelihoods on the innovations
    log_pdf = distribution.log_prob(e)

    ll = torch.sum(log_pdf - log_det, dim=1)

    return torch.mean(ll)


class MultivariateARCHModel:
    n: Optional[int]
    parameter_type: Type[Parameter]
    a: Optional[Parameter]
    b: Optional[Parameter]
    c: Optional[Parameter]
    d: Optional[Parameter]

    def __init__(
        self,
        constraint=ParameterConstraint.FULL,
        univariate_model: UnivariateScalingModel = UnivariateUnitScalingModel(),
        distribution: Distribution = NormalDistribution(),
        device: torch.device = None,
    ):
        self.constraint = constraint
        self.univariate_model = univariate_model
        self.distribution = distribution
        self.distribution.set_device(device)
        self.device = device

        # There should be a better way to do this.  Maybe add a set_device method.
        self.univariate_model.device = device

        self.n = self.a = self.b = self.c = self.d = None

        # The use of setattr here is to keep mypy happy.  It doeesn't like assigning
        # to attributes hinted as Callable.  It interprets them to be bound methods.
        if constraint == ParameterConstraint.SCALAR:
            self.parameter_type = ScalarParameter
        elif constraint == ParameterConstraint.DIAGONAL:
            self.parameter_type = DiagonalParameter
        elif constraint == ParameterConstraint.TRIANGULAR:
            self.parameter_type = TriangularParameter
        else:
            self.parameter_type = FullParameter

    def initialize_parameters(self, n: int) -> None:
        self.n = n
        # Initialize a and b as simple multiples of the identity
        self.a = self.parameter_type(n, 1.0 - constants.INITIAL_DECAY, self.device)
        self.b = self.parameter_type(n, constants.INITIAL_DECAY, self.device)
        self.c = self.parameter_type(n, 0.01, self.device)
        self.d = self.parameter_type(n, 1.0, self.device)
        self.log_parameters()

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, sample_scale: Any) -> None:
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(sample_scale, torch.Tensor):
            sample_scale = torch.tensor(
                sample_scale, dtype=torch.float, device=self.device
            )
        if (
            len(a.shape) != 2
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != sample_scale.shape
        ):
            raise ValueError(
                f"There must be two dimensions of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"sample_scale({sample_scale.shape}) that all agree"
            )

        self.n = a.shape[0]
        if self.n is not None:
            self.a = FullParameter(self.n)
            self.b = FullParameter(self.n)
            self.c = FullParameter(self.n)
            self.d = FullParameter(self.n)

            self.a.set(a)
            self.b.set(b)
            self.c.set(c)
            self.d.set(d)

        self.sample_scale = sample_scale

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "n": self.n,
            "sample_scale": self.sample_scale,
        }

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise ValueError("MultivariateARCHModel has not been initialized")
        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def log_parameters(self) -> None:
        if self.a and self.b and self.c and self.d:
            logging.info(
                "Multivariate ARCH model\n"
                f"a: {self.a.value.detach().numpy()},\n"
                f"b: {self.b.value.detach().numpy()},\n"
                f"c: {self.c.value.detach().numpy()},\n"
                f"d: {self.d.value.detach().numpy()}"
            )
        else:
            logging.info("Multivariate ARCH model has no initialized parameters")

    def _predict(
        self,
        observations: torch.Tensor,
        sample=False,
        scale_initial_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        lower triangular square roots of the sequence of covariance matrix estimates.

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols) of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` unit variance noise
                           rather than actual observations.
            initial_h: torch.Tensor - Initial covariance lower-triangular sqrt.
        Returns:
            h: torch.Tensor of predictions for each observation
            h_next: torch.Tensor prediction for next unobserved value
        """
        if scale_initial_value:
            if not isinstance(scale_initial_value, torch.Tensor):
                scale_initial_value = torch.tensor(
                    scale_initial_value, dtype=torch.float, device=self.device
                )
            scale_t = scale_initial_value
        else:
            scale_t = self.d @ self.sample_scale

        # We require ht to be lower traingular (even when parameters are full)
        # Ensure this using QR.
        scale_t_T = torch.linalg.qr(scale_t, mode="reduced")[1]
        scale_t = scale_t_T.T

        if constants.DEBUG:
            print(f"Initial scale: {scale_t}")
            print(f"self.d: {self.d.value if self.d is not None else None}")
            print(f"self.sample_scale: {self.sample_scale}")
        scale_sequence = []

        for k, obs in enumerate(observations):
            # Store the current ht before predicting next one
            scale_sequence.append(scale_t)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.

            a_scale_t = torch.clamp(
                self.a @ scale_t, min=constants.MIN_CLAMP, max=constants.MAX_CLAMP
            )

            if sample:
                # obs is noise that must be scaled
                obs = scale_t @ obs

            b_o = (self.b @ obs).unsqueeze(1)
            c_hbar = self.c @ self.sample_scale

            # The covariance is a_ht @ a_ht.T + b_o @ b_o.T + (c @ sample_scale) @ (c @ sample_scale).T
            # Unnecessary squaring is discouraged for nunerical stability.
            # Instead, we use only square roots and never explicity
            # compute the covariance.  This is a common 'trick' achieved
            # by concatenating the square roots in a larger array and
            # computing the QR factoriation, which computes the square
            # root of the sum of squares.  The covariance matrix isn't
            # formed explicitly in this code except at the very end when
            # it's time to return the covariance matrices to the user.

            m = torch.cat((a_scale_t, b_o, c_hbar), dim=1)

            # Unfortunately there's no QL factorization in PyTorch so we
            # transpose m and use the QR.  We only need the 'R' return
            # value, so the Q return value is dropped.

            scale_t_T = torch.linalg.qr(m.T, mode="reduced")[1]

            # Transpose ht to get the lower triangular version.

            scale_t = make_diagonal_nonnegative(scale_t_T.T)

        scale = torch.stack(scale_sequence)
        return scale, scale_t

    def __mean_log_likelihood(
        self,
        centered_observations: torch.Tensor,
        uv_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        This computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        # We pass uv_scale into this function rather than computing it
        # here because __mean_log_likelihood() is called in a training
        # loop and the univariate parameters are held constant while
        # the multivariate parameters are trained.  In other words,
        # uv_scale is constant through the optimization and we'd be
        # computing it on every iteration if we computed it here.
        # Similarly _predict() doesn't know about the univariate model
        # since it sees only scaled observations.  Clients should use only
        # mean_log_likelihood() and predict() which are more intuitive.

        scaled_centered_observations = centered_observations / uv_scale

        mv_scale = self._predict(scaled_centered_observations)[0]

        # It's important to use non-scaled observations in likelihood function
        mean_ll = joint_conditional_log_likelihood(
            centered_observations,
            mv_scale=mv_scale,
            uv_scale=uv_scale,
            distribution=self.distribution.get_instance(),
        )

        return mean_ll

    @torch.no_grad()
    def mean_log_likelihood(self, observations: torch.Tensor) -> float:
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).

        Arguments:
            observations: torch.Tensor of shape (n_obs, n_symbols)

        Return value:
            float - mean (per sample) log likelihood

        """
        uv_scale, uv_mean = self.univariate_model.predict(observations)[:2]
        centered_observations = observations - uv_mean
        result = self.__mean_log_likelihood(centered_observations, uv_scale)

        return float(result)

    def fit(self, observations: torch.Tensor) -> None:
        self.univariate_model.fit(observations)
        uv_scale, uv_mean = self.univariate_model.predict(observations)[:2]

        n = observations.shape[1]
        self.initialize_parameters(n)

        centered_observations = observations - uv_mean
        unscaled_centered_observations = centered_observations / uv_scale

        self.sample_scale = (
            torch.linalg.qr(unscaled_centered_observations, mode="reduced")[1]
        ).T / torch.sqrt(torch.tensor(unscaled_centered_observations.shape[0]))
        self.sample_scale = make_diagonal_nonnegative(self.sample_scale)
        logging.info(f"sample_scale:\n{self.sample_scale}")

        parameters = (
            self.get_optimizable_parameters()
            + self.distribution.get_optimizable_parameters()
        )

        optim = torch.optim.LBFGS(
            parameters,
            max_iter=constants.PROGRESS_ITERATIONS,
            lr=constants.LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure() -> float:
            safe_value = lambda x: x.value if x is not None else None
            if constants.DEBUG:
                print(f"a: {safe_value(self.a)}")
                print(f"b: {safe_value(self.b)}")
                print(f"c: {safe_value(self.c)}")
                print(f"d: {safe_value(self.d)}")
                print()
            optim.zero_grad()

            # Do not use scaled observations here; centering is okay.
            loss = -self.__mean_log_likelihood(centered_observations, uv_scale=uv_scale)
            loss.backward()

            return float(loss)

        optimize(optim, loss_closure, "multivariate model")

        self.distribution.log_parameters()
        self.log_parameters()

        logging.debug("Gradients: ")
        safe_grad = lambda x: x.value.grad if x is not None else None
        logging.debug(f"a.grad:\n{safe_grad(self.a)}")
        logging.debug(f"b.grad:\n{safe_grad(self.b)}")
        logging.debug(f"c.grad:\n{safe_grad(self.c)}")
        logging.debug(f"d.grad:\n{safe_grad(self.d)}")

    @torch.no_grad()
    def predict(
        self,
        observations: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        (
            uv_scale,
            uv_mean,
            uv_scale_next,
            uv_mean_next,
        ) = self.univariate_model.predict(observations)
        centered_observations = observations - uv_mean
        scaled_centered_observations = centered_observations / uv_scale

        mv_scale, mv_scale_next = self._predict(scaled_centered_observations)

        return mv_scale, uv_scale, uv_mean, mv_scale_next, uv_scale_next, uv_mean_next

    @torch.no_grad()
    def sample(
        self,
        n: Union[torch.Tensor, int],
        initial_mv_scale: Any,
        initial_uv_scale: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a random sampled output from the model.
        Arguments:
            n: torch.Tensor - Noise to use as input or
               int - Number of points to generate, in which case GWN is used.
            initial_h: torch.Tensor - Initial condition for sqrt of
                       covariance matrix (or correlation matrix when
                       internal univariate model is used)
            initial_sigma: torch.Tensor - Initial sigma for internal
                            univariate model if one is used
        Returns:
            output: torch.Tensor - Sample model output
            h: torch.Tensor - Sqrt of covariance used to scale the sample
        """
        if self.n is None:
            raise ValueError(
                "MultivariateARCHModel has not been trained or initialized"
            )

        if isinstance(n, int):
            n = self.distribution.get_instance().sample((n, self.n))

        # Next line is to keep mypy happy.
        if not isinstance(n, torch.Tensor):
            raise Exception("n isn't a tensor")

        mv_scale = self._predict(n, sample=True, scale_initial_value=initial_mv_scale)[
            0
        ]
        mv_scaled_noise = (mv_scale @ n.unsqueeze(2)).squeeze(2)

        output, uv_scale, uv_mean = self.univariate_model.sample(
            mv_scaled_noise, initial_uv_scale
        )

        return output, mv_scale, uv_scale, uv_mean


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        force=True,
    )
    multivariate_model = MultivariateARCHModel(
        constraint=ParameterConstraint.TRIANGULAR
    )
    multivariate_model.set_parameters(
        a=[[0.92, 0.0, 0.0], [-0.03, 0.95, 0.0], [-0.04, -0.02, 0.97]],
        b=[[0.4, 0.0, 0.0], [0.1, 0.3, 0.0], [0.13, 0.08, 0.2]],
        c=[[0.07, 0.0, 0.0], [0.04, 0.1, 0.0], [0.05, 0.005, 0.08]],
        d=[[1.0, 0.0, 0.0], [0.1, 0.6, 0.0], [-1.2, -0.8, 2]],
        sample_scale=[
            [0.008, 0.0, 0.0],
            [0.008, 0.01, 0.0],
            [0.008, 0.009, 0.005],
        ],
    )
    multivariate_model.univariate_model.set_parameters(n=3)

    mv_x, mv_scale, uv_scale, mu = multivariate_model.sample(
        10000, [[0.008, 0.0, 0.0], [0.008, 0.01, 0.0], [0.008, 0.009, 0.005]]
    )
    multivariate_model.fit(mv_x)
