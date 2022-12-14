# Standard Python
from abc import abstractmethod
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Common packages
import torch

# Local modules
from . import constants
from .distributions import Distribution, NormalDistribution
from .mean_models import MeanModel, ZeroMeanModel
from .parameters import (
    Parameter,
    DiagonalParameter,
)
from .optimize import optimize
from .util import to_tensor


def marginal_conditional_log_likelihood(
    centered_observations: torch.Tensor,
    scale: torch.Tensor,
    distribution: torch.distributions.Distribution,
) -> torch.Tensor:
    """
    Arguments:
       centered_observations: torch.Tensor of shape (n_obs, n_symbols)
       scale: torch.Tensor of shape (n_obs, n_symbols)
           Contains the estimated univariate (marginal) standard deviation for each observation.
       distribution: torch.distributions.distribution.Distribution instance
            which should have a log_prob() method.

       Note we assume distrubution was constructed with center=0 and shape=1.
       Any normalizing and recentering is achieved by explicit`transformations` here.

    Returns the mean log_likelihood"""

    scaled_observations = centered_observations / scale
    logging.debug(f"scaled_observations: \n{scaled_observations}")

    # Compute the log likelihoods on the innovations
    ll = distribution.log_prob(scaled_observations) - torch.log(scale)

    # For consistency with the multivariate case, we *sum* over the
    # variables (columns) first and *average* over the rows (observations).
    # Summing over the variables is equivalent to multiplying the
    # marginal distributions to get a join distribution.

    return torch.mean(torch.sum(ll, dim=1))


class UnivariateScalingModel(Protocol):
    device: Optional[torch.device]
    distribution: Distribution
    mean_model: MeanModel

    @abstractmethod
    def initialize_parameters(self, observations: torch.Tensor) -> None:
        """Abstract method with no implementation."""

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Abstract method with no implementation."""

    @property
    @abstractmethod
    def dimension(self) -> Optional[int]:
        """Abstract method with no implementation."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Abstract method with no implementation."""

    @property
    @abstractmethod
    def is_optimizable(self) -> bool:
        """Abstract method with no implementation."""

    @abstractmethod
    def log_parameters(self) -> None:
        """Abstract method with no implementation."""

    @abstractmethod
    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        """
        Arguments: None
        Returns a list of parameters that can be used in a
         `torch.optim.Optimizer` constructor.
        """

    @abstractmethod
    def _predict(
        self,
        centered_observations: torch.Tensor,
        sample=False,
        scale_initial_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            centered_observations: torch.Tensor of dimension (n_obs, n_symbols)
                                   of centered observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` is zero-mean, unit-variance noise
                           rather than actual observations.
            initial_scale: torch.Tensor - Initial condition for scale
            initial_mean: torch.Tensor - Initial condition for mean
        Returns:
            scale_next: torch.Tensor scale prediction for next unobserved value
            scale: torch.Tensor of scale predictions for each observation

        """

    def fit(self, observations: torch.Tensor) -> None:
        observations = to_tensor(observations, device=self.device)

        self.mean_model.initialize_parameters(observations)
        self.initialize_parameters(observations)

        model_parameters = self.get_optimizable_parameters()
        if len(model_parameters) == 0:
            raise ValueError(
                f"Mean model: Don't call fit() on a univariate model without tunable parameters. "
                f"If you're trying to fit the underlying mean model, use the univariate model "
                f"in a multivariate model and call fit() on the multivariate model."
            )

        model_parameters.extend(self.mean_model.get_optimizable_parameters())

        optim = torch.optim.LBFGS(
            model_parameters + self.distribution.get_optimizable_parameters(),
            max_iter=constants.PROGRESS_ITERATIONS,
            lr=constants.LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure() -> float:
            if constants.DEBUG:  # pragma: no cover
                self.log_parameters()
            optim.zero_grad()
            loss = -self.__mean_log_likelihood(observations)
            loss.backward()
            return float(loss)

        optimize(optim, loss_closure, "univariate model")

        self.distribution.log_parameters()
        self.mean_model.log_parameters()
        self.log_parameters()

    @torch.no_grad()
    def predict(
        self,
        observations: torch.Tensor,
        scale_initial_value: Optional[torch.Tensor] = None,
        mean_initial_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        observations = to_tensor(observations, device=self.device)
        mu_next, mu = self.mean_model.predict(observations, mean_initial_value)
        scale_next, scale = self._predict(observations - mu, scale_initial_value)

        return scale_next, mu_next, scale, mu

    def __mean_log_likelihood(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute and return the mean (per-sample) log likelihood (the total
        log likelihood divided by the number of samples).

        """
        mu = self.mean_model._predict(observations)[1]
        centered_observations = observations - mu
        scale = self._predict(centered_observations)[1]
        mean_ll = marginal_conditional_log_likelihood(
            centered_observations, scale, self.distribution.get_instance()
        )
        return mean_ll

    @torch.no_grad()
    def mean_log_likelihood(self, observations: torch.Tensor) -> float:
        """This is the inference version of mean_log_likelihood(), which is
        the version clients would normally use.  It computes the mean
        per-sample log likelihood (the total log likelihood divided by
        the number of samples).

        """
        observations = to_tensor(observations, device=self.device)
        return float(self.__mean_log_likelihood(observations))

    @torch.no_grad()
    def sample(
        self,
        n: Any,
        mean_initial_value: Any = None,
        scale_initial_value: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a random sampled output from the model.
        Arguments:
            n: torch.Tensor - CENTERED, UNSCALED (but possibly correlated)
                              noise to used as input or
                        int - Number of points to generate, in which case
                              GWN is generated here and used.
        Returns:
            output: torch.Tensor - Sample model output
            scale: torch.Tensor - Scale value used to produce the sample
            mean: torch.Tensor - Mean value used to produce the sample
        """

        if isinstance(n, int):
            dim = self.dimension
            if not isinstance(dim, int):
                raise ValueError("Model has not been trained/initialized")
            n = self.distribution.get_instance().sample((n, dim))

        scale = self._predict(n, sample=True, scale_initial_value=scale_initial_value)[
            1
        ]
        scaled_noise = scale * n
        mu = self.mean_model._predict(
            scaled_noise, sample=True, mean_initial_value=mean_initial_value
        )[1]
        output = scaled_noise + mu
        return output, scale, mu

    @torch.no_grad()
    def simulate(self, observations: Any, periods: int, samples: Optional[int] = None):
        """
        Performs a Monte Carlo simulation by drawing `samples` samples from the modeo
        for the next `periods` time periods.
        Arguments:
            observations: Any - Observations used to determine initial state for
                                the simulation.  The simulation will simulate the
                                conditions immediately following the observations.
            periods: int - Number of time periods per simulation
            samples: int - Number of simulations to perform (one if not provided)
        Returns:
            output: torch.Tensor - Shape (samples, periods, dimension) containing simulated outputs
            scale: torch.Tensor - Shape (samples, periods, dimension) containing univariate
                                  scaling (e.g., sqrt of the variance)
            mean: torch.Tensor - Shape (samples, periods, dimension) containing mean

        Note: One simulation is generated and the `samples` dimension of the output is
              dropped if samples == None

        """
        observations = to_tensor(observations)
        initial_scale_state, initial_mean_state = self.predict(observations)[:2]

        output_list = []
        scale_list = []
        mean_list = []

        for _ in range(samples if samples is not None else 1):
            output, scale, mean = self.sample(
                periods,
                scale_initial_value=initial_scale_state,
                mean_initial_value=initial_mean_state,
            )

            output_list.append(output)
            scale_list.append(scale)
            mean_list.append(mean)

            output = torch.stack(output_list, dim=0)
            scale = torch.stack(scale_list, dim=0)
            mean = torch.stack(mean_list, dim=0)

        if samples is None:
            output = output.squeeze(0)
            scale = scale.squeeze(0)
            mean = mean.squeeze(0)

        return output, scale, mean


class UnivariateUnitScalingModel(UnivariateScalingModel):
    distribution: Distribution
    mean_model: MeanModel

    device: Optional[torch.device]

    __dim: Optional[int]

    def __init__(
        self,
        distribution: Distribution = NormalDistribution(),
        device: Optional[torch.device] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        self.device = device
        self.distribution = distribution
        self.distribution.set_device(device)
        self.mean_model = mean_model
        self.__dim = None

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        self.__dim = observations.shape[1]

    def set_parameters(self, **kwargs) -> None:
        # Require that `dim` be passed since there's no way to infer it.
        self.__dim = kwargs["dim"]

    @property
    def dimension(self) -> Optional[int]:
        return self.__dim

    def get_parameters(self) -> Dict[str, Any]:
        return {"dim": self.__dim}

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return []

    @property
    def is_optimizable(self) -> bool:
        return False

    def log_parameters(self) -> None:
        pass

    def _predict(
        self,
        centered_observations: torch.Tensor,
        sample=False,
        scale_initial_value=None,
        mean_initial_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            centered_observations: torch.Tensor of dimension (n_obs, n_symbols)
                                   of centered observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` is zero-mean, unit-variance noise
                           rather than actual observations.
            scale_initial_value: torch.Tensor - Initial condition for scale (ignored)
            mean_initial_value: torch.Tensor - Initial condition for mean (ignored)

        Returns:
            scale_next: torch.Tensor scale prediction for next unobserved value
            scale: torch.Tensor of scale predictions for each observation

        """
        # Set all of the scaling to ones.
        scale = torch.ones(
            centered_observations.shape, dtype=torch.float, device=self.device
        )
        next_scale = torch.ones(
            centered_observations.shape[1], dtype=torch.float, device=self.device
        )
        return next_scale, scale


class UnivariateARCHModel(UnivariateScalingModel):
    a: Optional[Parameter]
    b: Optional[Parameter]
    c: Optional[Parameter]
    d: Optional[Parameter]
    sample_scale: Optional[torch.Tensor]

    __dim: Optional[int]

    def __init__(
        self,
        distribution: Distribution = NormalDistribution(),
        device: Optional[torch.device] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        self.__dim = self.a = self.b = self.c = self.d = None
        self.sample_scale = None
        self.distribution = distribution
        self.distribution.set_device(device)
        self.device = device
        self.mean_model = mean_model

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        dim = observations.shape[1]
        self.a = DiagonalParameter(
            dim, 1.0 - constants.INITIAL_DECAY, device=self.device
        )
        self.b = DiagonalParameter(dim, constants.INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(dim, 1.0, device=self.device)
        self.d = DiagonalParameter(dim, 1.0, device=self.device)
        self.sample_scale = torch.std(observations, dim=0)
        self.__dim = dim

    def set_parameters(self, **kwargs: Any) -> None:
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        sample_scale = kwargs["sample_scale"]

        a = to_tensor(a, device=self.device, requires_grad=True)
        b = to_tensor(b, device=self.device, requires_grad=True)
        c = to_tensor(c, device=self.device, requires_grad=True)
        d = to_tensor(d, device=self.device, requires_grad=True)
        sample_scale = to_tensor(sample_scale, device=self.device)

        if (
            len(a.shape) != 1
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != sample_scale.shape
        ):
            raise ValueError(
                f"The shapes of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"sample_scale({sample_scale.shape}) must have "
                "only and only one dimension that's consistent"
            )

        dim = a.shape[0]
        self.a = DiagonalParameter(dim).set(a)
        self.b = DiagonalParameter(dim).set(b)
        self.c = DiagonalParameter(dim).set(c)
        self.d = DiagonalParameter(dim).set(d)

        self.sample_scale = sample_scale

        self.__dim = dim

    @property
    def dimension(self) -> Optional[int]:
        return self.__dim

    def get_parameters(self) -> Dict[str, Any]:
        safe_value = lambda x: x.value.detach().numpy() if x is not None else None
        return {
            "a": safe_value(self.a),
            "b": safe_value(self.b),
            "c": safe_value(self.c),
            "d": safe_value(self.d),
            "sample_scale": self.sample_scale,
        }

    @property
    def is_optimizable(self) -> bool:
        return True

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise RuntimeError("UnivariateARCHModel has not been trained/initialized")

        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def log_parameters(self) -> None:
        if self.a and self.b and self.c and self.d and self.sample_scale is not None:
            logging.info(
                f"Univariate ARCH model parameters:\n"
                f"a: {self.a.value.detach().numpy()}, "
                f"b: {self.b.value.detach().numpy()}, "
                f"c: {self.c.value.detach().numpy()}, "
                f"d: {self.d.value.detach().numpy()}, "
                f"sample_scale: {self.sample_scale.numpy()}"
            )
        else:
            logging.info("Univariate variance model has no initialized parameters.")

    def _predict(
        self,
        centered_observations: torch.Tensor,
        sample=False,
        scale_initial_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and centered_observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            centered_observations: torch.Tensor of dimension (n_obs, n_symbols)
                                  of centered observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `centered_observations` unit variance noise
                           rather than actual observations.
            scale_initial_value: torch.Tensor - Initial value for scale.
        Returns:
            scale: torch.Tensor of predictions for each observation
            next_scale: torch.Tensor prediction for next unobserved value

        """
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise RuntimeError("Model has not been fit()")

        if scale_initial_value is not None:
            scale_t = scale_initial_value
        else:
            scale_t = self.d @ self.sample_scale  # type: ignore

        # mu, mu_next = self.mean_model._predict(centered_observations)
        # centered_observations = observations - mu

        scale_t = torch.maximum(scale_t, torch.tensor(float(constants.EPS)))
        scale_sequence = []

        for obs in centered_observations:
            # Store the current ht before predicting next one
            scale_sequence.append(scale_t)

            # The variance is (a * sigma)**2 + (b * o)**2 + (c * sample_scale)**2
            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_sigma = torch.clamp(
                self.a @ scale_t, min=constants.MIN_CLAMP, max=constants.MAX_CLAMP
            )

            if sample:
                # obs is noise that must be scaled
                obs = scale_t * obs

            b_o = self.b @ obs
            c_sample_scale = self.c @ self.sample_scale  # type: ignore

            # To avoid numerical issues associated with expressions of the form
            # sqrt(a**2 + b**2 + c**2), we use a similar trick as for the multivariate
            # case, which is to stack the variables (a, b, c) vertically and take
            # the column norms.  We depend on the vector_norm()
            # implementation being stable.

            m = torch.stack((a_sigma, b_o, c_sample_scale), dim=0)
            scale_t = torch.linalg.vector_norm(m, dim=0)

        scale = torch.stack(scale_sequence)
        return scale_t, scale


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        force=True,
    )
    # Example usage:
    univariate_model = UnivariateARCHModel()

    univariate_model.set_parameters(
        a=[0.80], b=[0.33], c=[0.5], d=[1.0], sample_scale=[0.0025]
    )

    # uv_x, uv_sigma = univariate_model.sample(50000, [0.01])[:2]
    # univariate_model.fit(uv_x)

    random_x = torch.randn((500, 1))
    random_x = random_x / torch.std(random_x) * 0.25
    univariate_model.fit(random_x)
    scale, _, scale_next = univariate_model.predict(random_x)[:3]
    print(scale[:10])
    print("...")
    print(scale[-10:])

    print(f"prediction: {scale_next}")
