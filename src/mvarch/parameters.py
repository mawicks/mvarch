# Standard Python
from abc import abstractmethod
import logging
from typing import Any, Optional, Protocol, Union
from enum import Enum

# Common packages
import torch

from . import matrix_ops


class ParameterConstraint(Enum):
    SCALAR = "scalar"
    DIAGONAL = "diagonal"
    TRIANGULAR = "triangular"
    FULL = "full"


class Parameter(Protocol):
    value: torch.Tensor

    @abstractmethod
    def __init__(self, n: int, scale: float, device: Optional[torch.device]):
        raise NotImplementedError

    @abstractmethod
    def _validate_value(value: torch.Tensor):
        raise NotImplementedError

    def set(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(
                value, device=self.device, dtype=torch.float, requires_grad=True
            )

        self._validate_value(value)
        self.value = value

    @abstractmethod
    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ScalarParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        self.device = device
        self.value = torch.tensor(scale, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> None:
        if len(value.shape) != 0:
            raise ValueError(f"Value isn't a scalar: {value}")

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            return self.value * other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class DiagonalParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        self.device = device
        self.value = scale * torch.ones(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> None:
        if len(value.shape) != 1:
            raise ValueError(f"Value isn't a vector: {value}")

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            # Because `other` is square, if we just use `*` here, there's ambiguity
            # whether we're operating on rows or columns (should be rows).
            # Use unsqueeze() and expand() to disambiguate what should happen.
            return self.value.unsqueeze(1).expand(other.shape) * other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class TriangularParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> None:
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"Value isn't a square matrix: {value}")

        if not matrix_ops.is_lower_triangular(value):
            raise ValueError(f"Value isn't lower triangular: {value}")

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        # self.value was initialized to be triangular, so the
        # torch.tril() below may seem unnecessary.  Using the torch.tril()
        # ensures that the upper entries remain excluded from gradient
        # calculations and don't get updated by the optimizer.
        try:
            return torch.tril(self.value) @ other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class FullParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> None:
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"Value isn't a square matrix: {value}")

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            return self.value @ other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


if __name__ == "__main__":

    t = TriangularParameter(3, 10.0)
    print(t.value)
