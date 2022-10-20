# Standard Python
from abc import abstractmethod
import logging
from typing import Any, Dict, List, Protocol


# Common packages
import torch


class Distribution(Protocol):
    device: torch.device

    @abstractmethod
    def set_parameters(self, **kwargs: Any) -> None:
        """Abstract method with no implementation."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Abstract method with no implementation."""

    @abstractmethod
    def std_dev(self) -> float:
        """Abstract method with no implementation."""

    @abstractmethod
    def log_parameters(self) -> None:
        """Abstract method with no implementation."""

    @abstractmethod
    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        """Abstract method with no implementation."""

    @abstractmethod
    def get_instance(self) -> torch.distributions.Distribution:
        """Abstract method with no implementation."""

    def set_device(self, device) -> None:
        self.device = device


class NormalDistribution(Distribution):
    def __init__(self, device=None):
        self.device = device
        self.instance = torch.distributions.normal.Normal(
            loc=torch.tensor(0.0, dtype=torch.float, device=device),
            scale=torch.tensor(1.0, dtype=torch.float, device=device),
        )

    def set_parameters(self, **kwargs: Any) -> None:
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def std_dev(self) -> float:
        return 1.0

    def log_parameters(self) -> None:
        return

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return []

    def get_instance(self) -> torch.distributions.Distribution:
        return self.instance


class StudentTDistribution(Distribution):
    def __init__(self, device=None):
        self.device = device
        self.df = torch.tensor(
            6.0, dtype=torch.float, device=device, requires_grad=True
        )

    def set_parameters(self, **kwargs: Any) -> None:
        df = kwargs["df"]

        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df, dtype=torch.float, device=self.device)
        df.requires_grad = True
        self.df = df

    def get_parameters(self) -> Dict[str, Any]:
        return {"df": self.df}

    def std_dev(self) -> float:
        if torch.abs(self.df) > 2:
            return float(torch.sqrt(torch.abs(self.df) / torch.abs(self.df - 2)))
        elif self.df > 1:
            return float("inf")
        else:
            return float("nan")

    def log_parameters(self) -> None:
        logging.info(f"StudentT DF: {self.df:.3f}")

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return [self.df]

    def get_instance(self) -> torch.distributions.Distribution:
        return torch.distributions.studentT.StudentT(torch.abs(self.df))


if __name__ == "__main__":  # pragma: no cover
    n = NormalDistribution()
    t = StudentTDistribution()
