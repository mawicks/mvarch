# Standard Python
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Common packages
import torch

# Local modules
from . import constants

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
)
from .matrix_ops import make_diagonal_nonnegative
from .optimize import optimize


def joint_conditional_log_likelihood(
    centered_observations: torch.Tensor,
    mv_scale: torch.Tensor,
    distribution: torch.distributions.Distribution,
    uv_scale=Union[torch.Tensor, None],
) -> torch.Tensor:
    """
    Arguments:
       centered_observations: torch.Tensor of shape (n_obs, n_symbols)
       mv_scale: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.Distribution or other object with a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1
           Any normalizing and recentering is achieved by explicit`transformations` here.
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

    log_diag = torch.log(torch.abs(torch.diagonal(mv_scale, dim1=1, dim2=2)))

    if uv_scale is not None:
        log_diag = log_diag + torch.log(torch.abs(uv_scale))
        centered_observations = centered_observations / uv_scale

    # First get the innovations sequence by forming transformation^(-1)*centered_observations
    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `t` is (n_obs, n, n) while
    # the shape of `centered_observations` is (n_obs, n). Without adding the
    # extract dimension to observations, the solver won't see conforming dimensions.
    # We remove the ambiguity, by making observations` have shape (n_obj, n, 1), then
    # we remove the extra dimension from e.
    e = torch.linalg.solve_triangular(
        mv_scale,
        centered_observations.unsqueeze(2),
        upper=False,
    ).squeeze(2)

    logging.debug(f"e: \n{e}")

    # Compute the log likelihoods on the innovations
    log_pdf = distribution.log_prob(e)

    # Sum across the symbols (columns)
    ll = torch.sum(log_pdf - log_diag, dim=1)

    # Mean across the rows (samples)
    return torch.mean(ll)


class MultivariateARCHModel:
    parameter_type: Type[Parameter]

    a: Optional[Parameter]
    b: Optional[Parameter]
    c: Optional[Parameter]
    d: Optional[Parameter]

    def __init__(
        self,
        constraint=ParameterConstraint.FULL,
        univariate_model: UnivariateScalingModel = UnivariateUnitScalingModel(),
        device: torch.device = None,
    ):
        self.constraint = constraint
        self.univariate_model = univariate_model

        self.distribution = univariate_model.distribution

        self.distribution.set_device(device)
        self.device = device

        # There should be a better way to do this.  Maybe add a set_device method.
        self.univariate_model.device = device

        self.a = self.b = self.c = self.d = None

        if constraint == ParameterConstraint.SCALAR:
            self.parameter_type = ScalarParameter
        elif constraint == ParameterConstraint.DIAGONAL:
            self.parameter_type = DiagonalParameter
        elif constraint == ParameterConstraint.TRIANGULAR:
            self.parameter_type = TriangularParameter
        else:
            self.parameter_type = FullParameter

    def initialize_parameters(
        self, unscaled_centered_observations: torch.Tensor
    ) -> None:
        n = unscaled_centered_observations.shape[1]
        # Initialize a and b as simple multiples of the identity
        self.a = self.parameter_type(n, 1.0 - constants.INITIAL_DECAY, self.device)
        self.b = self.parameter_type(n, constants.INITIAL_DECAY, self.device)
        self.c = self.parameter_type(n, 0.01, self.device)
        self.d = self.parameter_type(n, 1.0, self.device)

        # Recenter. This is important when the centering in the caller
        # was done with initialized but untuned means.
        unscaled_centered_observations = unscaled_centered_observations - torch.mean(
            unscaled_centered_observations, dim=0
        )

        # Pre-multiply by the sqrt of the sample size.
        # This is equivalent to dividing o.T@o byn
        # C = E[o.T @ o] approximated by (o.T @o)/n
        # The qr in transform_matrix factors o.T @ o

        self.sample_scale = self.transform_matrix(
            unscaled_centered_observations.T
            / torch.sqrt(torch.tensor(unscaled_centered_observations.shape[0]))
        )

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, sample_scale: Any) -> None:
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(
                a, dtype=torch.float, device=self.device, requires_grad=True
            )
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(
                b, dtype=torch.float, device=self.device, requires_grad=True
            )
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(
                c, dtype=torch.float, device=self.device, requires_grad=True
            )
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(
                d, dtype=torch.float, device=self.device, requires_grad=True
            )
        if not isinstance(sample_scale, torch.Tensor):
            sample_scale = torch.tensor(
                sample_scale, dtype=torch.float, device=self.device
            )
        if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
            raise ValueError(
                f"There dimensions of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), and d({d.shape}) must agree."
            )
        if (
            len(sample_scale.shape) != 2
            or sample_scale.shape[0] != sample_scale.shape[1]
            or (len(a.shape) >= 1 and a.shape[0] != sample_scale.shape[0])
        ):
            raise ValueError(
                f"The shape of sample_scale ({sample_scale.shape}) must be square and must "
                f"conform with parameter shapes ({a.shape})"
            )

        n = a.shape[0]
        self.a = self.parameter_type(n).set(a)
        self.b = self.parameter_type(n).set(b)
        self.c = self.parameter_type(n).set(c)
        self.d = self.parameter_type(n).set(d)

        self.sample_scale = sample_scale

    def get_parameters(self) -> Dict[str, Any]:
        safe_value = lambda x: x.value.detach().numpy() if x is not None else None
        return {
            "a": safe_value(self.a),
            "b": safe_value(self.b),
            "c": safe_value(self.c),
            "d": safe_value(self.d),
            "sample_scale": self.sample_scale,
        }

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise RuntimeError("MultivariateARCHModel has not been initialized")
        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def log_parameters(self) -> None:
        if self.a and self.b and self.c and self.d:
            logging.info(
                "Multivariate ARCH model\n"
                f"a: {self.a.value.detach().numpy()},\n"
                f"b: {self.b.value.detach().numpy()},\n"
                f"c: {self.c.value.detach().numpy()},\n"
                f"d: {self.d.value.detach().numpy()}, \n"
                f"sample_scale: {self.sample_scale.numpy()}"
            )
        else:
            logging.info("Multivariate ARCH model has no initialized parameters")

    def transform_matrix(self, scale_matrix):
        # We require the scaling matrix to satisfy some constraints
        # such as being lower traingular with positive diagonal
        # entries (even when parameters are full).  This is similar to
        # requiring the standard deviation to be positive.  It's
        # transformed to an "equivalent" matrix satisfying the
        # necessary properties.

        # Unfortunately there's no QL factorization in PyTorch so we
        # transpose m and use the QR.  We only need the 'R' return
        # value, so the Q return value is dropped.

        upper = torch.linalg.qr(scale_matrix.T, mode="reduced")[1]

        if not isinstance(self.univariate_model, UnivariateUnitScalingModel):
            # Normalize the colums (while in upper triangular form)
            # so that the univariate model determines the variances
            # and the multivariate model determines the correlations.
            upper = torch.nn.functional.normalize(upper, dim=0)

        # Transpose to make lower triangular, then make diagonal positive.
        lower = make_diagonal_nonnegative(upper.T)
        return lower

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
            if (
                len(scale_initial_value.shape) != 2
                or scale_initial_value.shape[0] != scale_initial_value.shape[1]
            ):
                raise ValueError(
                    f"Shape of scale_intial_value ({scale_initial_value.shape}) must be square "
                )
            scale_t = scale_initial_value
        else:
            scale_t = self.d @ self.sample_scale

        # We require the inttial scale matrix satisfy some constraints
        # such as being lower traingular with positive diagonal entries
        scale_t = self.transform_matrix(scale_t)

        if constants.DEBUG:  # pragma: no cover
            print(f"Initial scale: {scale_t}")
            print(f"self.d: {self.d.value if self.d is not None else None}")
            print(f"self.sample_scale: {self.sample_scale}")
        scale_sequence = []

        for obs in observations:
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
            c_sample_scale = self.c @ self.sample_scale

            # The covariance is
            # a_ht @ a_ht.T + b_o @ b_o.T + (c @ sample_scale) @ (c @ sample_scale).T
            # Unnecessary squaring is discouraged for nunerical stability.
            # Instead, we use only square roots and never explicity
            # compute the covariance.  This is a common 'trick' achieved
            # by concatenating the square roots in a larger array and
            # computing the QR factoriation, which computes the square
            # root of the sum of squares.  The covariance matrix isn't
            # formed explicitly in this code except at the very end when
            # it's time to return the covariance matrices to the user.

            m = torch.cat((a_scale_t, b_o, c_sample_scale), dim=1)

            # Transform `m` to an equivalent matrix having required dimensions and properties
            scale_t = self.transform_matrix(m)

        scale = torch.stack(scale_sequence)
        return scale, scale_t

    def __mean_log_likelihood(
        self,
        centered_observations: torch.Tensor,
        uv_scale: torch.Tensor,
    ) -> torch.Tensor:
        """This computes the mean per-sample log likelihood (the total log
        likelihood divided by the number of samples).

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
        """This is the inference version of mean_log_likelihood(), which is
        the version clients would normally use.  It computes the mean
        per-sample log likelihood (the total log likelihood divided by
        the number of samples).

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
        """Fit a multivariate model along with any underlying univariate,
        mean, and distribution models.

        """
        centered_observations: Optional[torch.Tensor]
        uv_scale: Optional[torch.Tensor]
        uv_mean: Optional[torch.Tensor]

        # If the underlying univariate model has optimizeable
        # parameters, optimize it.  The fit() call also tunes the mean
        # model and the distribution model parameters contained within
        # the univariate model.

        if self.univariate_model.is_optimizable:
            self.univariate_model.fit(observations)
            uv_scale, uv_mean = self.univariate_model.predict(observations)[:2]
            centered_observations = observations - uv_mean
            self.initialize_parameters(centered_observations / uv_scale)
        else:
            # Since we don't call fit() on the univariate model, its
            # mean model will have be initialized. Initialize its mean
            # model directly.
            self.univariate_model.mean_model.initialize_parameters(observations)
            self.univariate_model.initialize_parameters(observations)
            self.initialize_parameters(observations)
            centered_observations = uv_scale = uv_mean = None

        self.log_parameters()

        # We always optimize the multivariate model parameters here.
        parameters = self.get_optimizable_parameters()

        # If the univariate model isn't optimizable alone, add the
        # mean model parameters and the distribution parameters,
        # because they weren't optimized above.

        if not self.univariate_model.is_optimizable:
            parameters = (
                parameters
                + self.univariate_model.mean_model.get_optimizable_parameters()
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
            if constants.DEBUG:  # pragma: no cover
                print(f"a: {safe_value(self.a)}")
                print(f"b: {safe_value(self.b)}")
                print(f"c: {safe_value(self.c)}")
                print(f"d: {safe_value(self.d)}")
                print()

            optim.zero_grad()

            # If the underlying univariate model is being optimized,
            # then the univariate predictions must be recomputed here.
            # Otherwise they come from above.
            if self.univariate_model.is_optimizable:
                assert isinstance(centered_observations, torch.Tensor)
                assert isinstance(uv_scale, torch.Tensor)

                closure_centered_observations = centered_observations
                closure_uv_scale = uv_scale
            else:
                closure_uv_mean = self.univariate_model.mean_model._predict(
                    observations
                )[0]
                closure_centered_observations = observations - closure_uv_mean
                closure_uv_scale = self.univariate_model._predict(
                    closure_centered_observations
                )[0]
                closure_centered_observations = (
                    closure_centered_observations / closure_uv_scale
                )

            # Do not use scaled observations here; centering is okay.
            loss = -self.__mean_log_likelihood(
                closure_centered_observations, uv_scale=closure_uv_scale
            )
            loss.backward()

            return float(loss)

        optimize(optim, loss_closure, "multivariate model")

        self.distribution.log_parameters()
        self.univariate_model.mean_model.log_parameters()
        self.univariate_model.log_parameters()
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
        mv_scale_initial_value: Any = None,
        uv_scale_initial_value: Any = None,
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
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise RuntimeError(
                "MultivariateARCHModel has not been trained or initialized"
            )

        if isinstance(n, int):
            n = self.distribution.get_instance().sample((n, self.sample_scale.shape[0]))

        # Next line is to keep mypy happy.
        assert isinstance(n, torch.Tensor)

        mv_scale = self._predict(
            n, sample=True, scale_initial_value=mv_scale_initial_value
        )[0]
        mv_scaled_noise = (mv_scale @ n.unsqueeze(2)).squeeze(2)

        output, uv_scale, uv_mean = self.univariate_model.sample(
            mv_scaled_noise, uv_scale_initial_value
        )

        return output, mv_scale, uv_scale, uv_mean


if __name__ == "__main__":  # pragma: no cover
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
