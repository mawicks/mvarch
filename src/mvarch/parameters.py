from __future__ import annotations

# Standard Python
from abc import abstractmethod
from typing import Any, Optional, Protocol
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
    device: Optional[torch.device]

    @abstractmethod
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        """Abstract method with no implementation."""

    @abstractmethod
    def _validate_value(self, value: torch.Tensor) -> Parameter:
        """Abstract method with no implementation."""

    def set(self, value: Any) -> Parameter:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(
                value, device=self.device, dtype=torch.float, requires_grad=True
            )

        self._validate_value(value)
        self.value = value
        return self

    @abstractmethod
    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Abstract method with no implementation."""


class ScalarParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        super().__init__(n, scale, device)
        self.device = device
        self.value = torch.tensor(scale, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> ScalarParameter:
        if len(value.shape) != 0:
            raise ValueError(f"Value isn't a scalar: {value}")
        return self

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.value * other


class DiagonalParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        super().__init__(n, scale, device)
        self.device = device
        self.value = scale * torch.ones(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> DiagonalParameter:
        if len(value.shape) != 1:
            raise ValueError(f"Value isn't a vector: {value}")
        return self

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            # If `other` is square and we blindly use `*` here, there's
            # ambiguity whether we're operating on rows or columns
            # (should be rows).  Use unsqueeze() and expand() to
            # disambiguate what should happen.
            if len(other.shape) > 1:
                return self.value.unsqueeze(1).expand(other.shape) * other
            else:
                return self.value * other

        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class TriangularParameter(Parameter):
    def __init__(
        self, n: int, scale: float = 1.0, device: Optional[torch.device] = None
    ):
        super().__init__(n, scale, device)
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> TriangularParameter:
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"Value isn't a square matrix: {value}")

        if not matrix_ops.is_lower_triangular(value):
            raise ValueError(f"Value isn't lower triangular: {value}")
        return self

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
        super().__init__(n, scale, device)
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def _validate_value(self, value: torch.Tensor) -> FullParameter:
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"Value isn't a square matrix: {value}")
        return self

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            return self.value @ other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


if __name__ == "__main__":  # pragma: no cover
    t = TriangularParameter(3, 10.0)
    print(t.value)
