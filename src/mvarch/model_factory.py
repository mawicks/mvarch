from typing import Any, Dict, Optional

from . import distributions
from . import mean_models
from . import parameters
from . import univariate_models
from . import multivariate_models

DISTRIBUTION_CHOICES = {
    "normal": distributions.NormalDistribution,
    "studentt": distributions.StudentTDistribution,
}

MEAN_CHOICES = {
    "zero": mean_models.ZeroMeanModel,
    "constant": mean_models.ConstantMeanModel,
    "arma": mean_models.ARMAMeanModel,
}

UNIVARIATE_CHOICES = {
    "arch": univariate_models.UnivariateARCHModel,
    "none": univariate_models.UnivariateUnitScalingModel,
}

MULTIVARIATE_CHOICES = {
    "mvarch": multivariate_models.MultivariateARCHModel,
    "none": None,
}

CONSTRAINT_CHOICES = {
    "scalar": parameters.ParameterConstraint.SCALAR,
    "diagonal": parameters.ParameterConstraint.DIAGONAL,
    "triangular": parameters.ParameterConstraint.TRIANGULAR,
    "none": parameters.ParameterConstraint.FULL,
}


def get_choice(name: str, value: Optional[str], dictionary: Dict[str, Any]):
    if value is None:
        value = "none"
    if value not in dictionary.keys():
        allowed_list = [f"'{key}'" for key in dictionary.keys()]
        allowed_string = ", ".join(allowed_list[:-1]) + ", or " + allowed_list[-1]
        raise ValueError(f"{name}='{value}': '{name}' must be {allowed_string}")

    return dictionary[value]


def model_factory(
    distribution: str = "normal",
    mean: str = "zero",
    univariate: str = "arch",
    constraint: str = "none",
    multivariate: str = "mvarch",
):
    distribution_type = get_choice("distribution", distribution, DISTRIBUTION_CHOICES)
    mean_type = get_choice("mean", mean, MEAN_CHOICES)
    univariate_type = get_choice("univariate", univariate, UNIVARIATE_CHOICES)
    multivariate_type = get_choice("multivariate", multivariate, MULTIVARIATE_CHOICES)
    constraint_type = get_choice("constraint", constraint, CONSTRAINT_CHOICES)

    univariate_model = univariate_type(
        distribution=distribution_type(), mean_model=mean_type()
    )

    if multivariate_type is not None:
        model = multivariate_type(
            univariate_model=univariate_model, constraint=constraint_type
        )
    else:
        model = univariate_model

    return model


if __name__ == "__main__":  # pragma: no cover
    model_factory()
