# Standard Python
from abc import abstractmethod
import logging
from typing import Any, Dict, List, Protocol, Union


# Common packages
import torch


class Distribution(Protocol):
    device: torch.device

    @abstractmethod
    def set_parameters(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_instance(self) -> torch.distributions.Distribution:
        raise NotImplementedError

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

    def log_parameters(self) -> None:
        return

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return []

    def get_instance(self) -> torch.distributions.Distribution:
        return self.instance


class StudentTDistribution(Distribution):
    def __init__(self, device=None):
        self.device = device
        self.dof = torch.tensor(
            6.0, dtype=torch.float, device=device, requires_grad=True
        )

    def set_parameters(self, **kwargs: Any) -> None:
        dof = kwargs["dof"]

        if not isinstance(dof, torch.Tensor):
            dof = torch.tensor(dof, dtype=torch.float, device=self.device)
        dof.requires_grad = True
        self.dof = dof

    def get_parameters(self) -> Dict[str, Any]:
        return {"dof": self.dof}

    def log_parameters(self) -> None:
        logging.info(f"StudentT DOF: {self.dof:.3f}")

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        print("dof: ", self.dof)

        return [self.dof]

    def get_instance(self) -> torch.distributions.Distribution:
        return torch.distributions.studentT.StudentT(self.dof)


if __name__ == "__main__":

    n = NormalDistribution()
    t = StudentTDistribution()
