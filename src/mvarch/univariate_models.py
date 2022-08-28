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


def marginal_conditional_log_likelihood(
    observations: torch.Tensor,
    scale: torch.Tensor,
    distribution: torch.distributions.Distribution,
) -> torch.Tensor:
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       scale: torch.Tensor of shape (n_obs, n_symbols)
           Contains the estimated univariate (marginal) standard deviation for each observation.
       distribution: torch.distributions.distribution.Distribution instance
            which should have a log_prob() method.

       Note we assume distrubution was constructed with center=0 and shape=1.
       Any normalizing and recentering is achieved by explicit`transformations` here.

    Returns the mean log_likelihood"""

    scaled_observations = observations / scale
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
    n: Optional[int]

    @abstractmethod
    def initialize_parameters(self, observations: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        """
        Arguments: None
        Returns a list of parameters that can be used in a
         `torch.optim.Optimizer` constructor.
        """
        raise NotImplementedError

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
            scale: torch.Tensor of scale predictions for each observation
            scale_next: torch.Tensor scale prediction for next unobserved value

        """
        raise NotImplementedError

    def fit(self, observations: torch.Tensor) -> None:
        self.distribution.log_parameters()

        self.mean_model.initialize_parameters(observations)
        self.mean_model.log_parameters()

        self.initialize_parameters(observations)
        self.log_parameters()

        model_parameters = (
            self.mean_model.get_optimizable_parameters()
            + self.get_optimizable_parameters()
        )
        if len(model_parameters) > 0:
            optim = torch.optim.LBFGS(
                model_parameters + self.distribution.get_optimizable_parameters(),
                max_iter=constants.PROGRESS_ITERATIONS,
                lr=constants.LEARNING_RATE,
                line_search_fn="strong_wolfe",
            )

            def loss_closure() -> float:
                if constants.DEBUG:
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
        mu, mu_next = self.mean_model.predict(observations, mean_initial_value)
        scale, scale_next = self._predict(observations - mu, scale_initial_value)

        return scale, mu, scale_next, mu_next

    def __mean_log_likelihood(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute and return the mean (per-sample) log likelihood (the total
        log likelihood divided by the number of samples).

        """
        mu = self.mean_model._predict(observations)[0]
        centered_observations = observations - mu
        scale = self._predict(centered_observations)[0]
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
            if not isinstance(self.n, int):
                raise ValueError("Model has not been trained/initialized")
            n = self.distribution.get_instance().sample((n, self.n))

        scale = self._predict(n, sample=True, scale_initial_value=scale_initial_value)[
            0
        ]
        scaled_noise = scale * n
        mu = self.mean_model._predict(
            scaled_noise, sample=True, mean_initial_value=mean_initial_value
        )[0]
        output = scaled_noise + mu
        return output, scale, mu


class UnivariateUnitScalingModel(UnivariateScalingModel):
    distribution: Distribution
    device: Optional[torch.device]
    mean_model: MeanModel
    n: Optional[int]

    def __init__(
        self,
        distribution: Distribution = NormalDistribution(),
        device: Optional[torch.device] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        if not isinstance(mean_model, ZeroMeanModel):
            raise ValueError(
                f"Mean model: {type(mean_model)} Don't use a mean model "
                " having parameters with a parameterless scaling model. "
                "Change the mean model to ZeroMeanModel() or change the "
                "univariate model to something with parameters."
            )
        self.device = device
        self.distribution = distribution
        self.distribution.set_device(device)
        self.mean_model = mean_model
        self.n = None

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        self.n = observations.shape[1]

    def set_parameters(self, **kwargs) -> None:
        # Require that `n` be passed since there's no way to infer it.
        self.n = kwargs["n"]

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return []

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
            scale: torch.Tensor of scale predictions for each observation
            scale_next: torch.Tensor scale prediction for next unobserved value

        """
        # Set all of the scaling to ones.
        scale = torch.ones(
            centered_observations.shape, dtype=torch.float, device=self.device
        )
        scale_next = torch.ones(
            centered_observations.shape[1], dtype=torch.float, device=self.device
        )
        return scale, scale_next


class UnivariateARCHModel(UnivariateScalingModel):
    a: Optional[Parameter]
    b: Optional[Parameter]
    c: Optional[Parameter]
    d: Optional[Parameter]
    sample_scale: Optional[torch.Tensor]

    n: Optional[int]

    def __init__(
        self,
        distribution: Distribution = NormalDistribution(),
        device: Optional[torch.device] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        self.a = self.b = self.c = self.d = None
        self.sample_scale = None
        self.distribution = distribution
        self.distribution.set_device(device)
        self.device = device
        self.mean_model = mean_model

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        n = observations.shape[1]
        self.a = DiagonalParameter(n, 1.0 - constants.INITIAL_DECAY, device=self.device)
        self.b = DiagonalParameter(n, constants.INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(n, 1.0, device=self.device)
        self.d = DiagonalParameter(n, 1.0, device=self.device)
        self.sample_scale = torch.std(observations, dim=0)
        self.n = n

    def set_parameters(self, **kwargs: Any) -> None:
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        sample_scale = kwargs["sample_scale"]

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

        n = a.shape[0]
        self.a = DiagonalParameter(n).set(a)
        self.b = DiagonalParameter(n).set(b)
        self.c = DiagonalParameter(n).set(c)
        self.d = DiagonalParameter(n).set(d)

        self.sample_scale = sample_scale

        self.n = n

    def get_parameters(self) -> Dict[str, Any]:
        safe_value = lambda x: x.value if x is not None else None
        return {
            "a": safe_value(self.a),
            "b": safe_value(self.b),
            "c": safe_value(self.c),
            "d": safe_value(self.d),
            "sample_scale": self.sample_scale,
        }

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise ValueError("UnivariateARCHModel has not been trained/initialized")

        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def log_parameters(self) -> None:
        if self.a and self.b and self.c and self.d and self.sample_scale is not None:
            logging.info(
                "Univariate variance model\n"
                f"a: {self.a.value.detach().numpy()}, "
                f"b: {self.b.value.detach().numpy()}, "
                f"c: {self.c.value.detach().numpy()}, "
                f"d: {self.d.value.detach().numpy()}, "
                f"sample_scale: {self.sample_scale.numpy()}"
            )
            logging.info(f"sample_scale:\n{self.sample_scale}")
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
            scale_next: torch.Tensor prediction for next unobserved value

        """
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise Exception("Model has not been fit()")

        if scale_initial_value:
            if not isinstance(scale_initial_value, torch.Tensor):
                scale_initial_value = torch.tensor(
                    scale_initial_value, dtype=torch.float, device=self.device
                )
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
        return scale, scale_t


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        force=True,
    )
    # Example usage:
    univariate_model = UnivariateARCHModel()

    univariate_model.set_parameters(
        a=[0.90], b=[0.33], c=[0.25], d=[1.0], sample_scale=[0.01]
    )
    uv_x, uv_sigma = univariate_model.sample(10000, [0.01])[:2]
    univariate_model.fit(uv_x)
