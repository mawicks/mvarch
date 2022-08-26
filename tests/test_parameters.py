import pytest

import torch

# Local modules
from mvarch.parameters import (
    Parameter,
    ScalarParameter,
    DiagonalParameter,
    TriangularParameter,
    FullParameter,
)


@pytest.mark.parametrize(
    "parameter_type, value, expect_value_error, other, mismatch, expected",
    [
        #
        # ScalarParameter test cases
        # Valid multiplication by another matrix
        (
            ScalarParameter,
            2.0,
            False,
            [[1, 2], [3, 9]],
            False,
            [[2, 4], [6, 18]],
        ),
        # Valid multiplication by another vector
        (ScalarParameter, 2.0, False, [1, 2], False, [2, 4]),
        # Invalid vector parameter
        (ScalarParameter, [1, 2], True, None, False, None),
        # Invalid matrix parameter
        (ScalarParameter, [[1, 2], [3, 4]], True, None, False, None),
        #
        # DiagonalParameter test cases
        # Valid multiplication by another matrix
        (
            DiagonalParameter,
            [5, 7],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [21, 63]],
        ),
        # Valid multiplication by another vector
        (
            DiagonalParameter,
            [5, 7],
            False,
            [1, 2],
            False,
            [5, 14],
        ),
        # Other matrix doesn't conform
        (
            DiagonalParameter,
            [5, 7],
            False,
            [[1, 2], [3, 4], [5, 6]],
            True,
            None,
        ),
        # Scalar parameter isn't valid
        (DiagonalParameter, 2.0, True, None, False, None),
        # Matrix parameter isn't valid
        (DiagonalParameter, [[1, 2], [3, 4]], True, None, False, None),
        #
        # TriangularParameter test cases
        # Valid multiplication by another matrix
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [40, 113]],
        ),
        # Valid multiplication by another vector
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [1, 2],
            False,
            [5, 29],
        ),
        # Other matrix doesn't conform
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 4], [5, 6]],
            True,
            None,
        ),
        # Invalid scalar parameter
        (TriangularParameter, 2.0, True, None, False, None),
        # Invalid vector parameter
        (TriangularParameter, [1, 2], True, None, False, None),
        # Invalid full matrix parameter
        (
            TriangularParameter,
            [[1, 2], [3, 4]],
            True,
            None,
            False,
            None,
        ),  # Not triangular
        #
        # FullParameter test cases
        # Valid mulitplication by another matrix (lower triangular parameter)
        (
            FullParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [40, 113]],
        ),  # Other matrix
        # Valid mulitplication by another matrix
        (
            FullParameter,
            [[5, 7], [11, 13]],
            False,
            [[1, 2], [3, 9]],
            False,
            [
                [26, 73],
                [50, 139],
            ],
        ),
        # Valid multiplication by another vector
        (
            FullParameter,
            [[5, 7], [11, 13]],
            False,
            [1, 2],
            False,
            [19, 37],
        ),
        # Other matrix doesn't conform
        (
            FullParameter,
            [[5, 7], [11, 13]],
            False,
            [[1, 2], [3, 4], [5, 6]],
            True,
            [
                [26, 73],
                [50, 139],
            ],
        ),
        # Scalar parameter isn't valid
        (FullParameter, 2.0, True, None, False, None),
        # Vector parameter isn't valid
        (FullParameter, [1, 2], True, None, False, None),  # Vector not allowed
    ],
)
def test_scalar_parameter(
    parameter_type, value, expect_value_error, other, mismatch, expected
):
    # Test construction
    parameter = parameter_type(n=2, scale=1.0)

    if expect_value_error:
        with pytest.raises(ValueError):
            print(f"{value}")
            parameter.set(value)

    else:
        parameter.set(value)

        # Confirm that __matmul__() works using the @ operator
        other = torch.tensor(other, dtype=torch.float)

        if mismatch:
            with pytest.raises(RuntimeError):
                result = parameter @ other
        else:
            result = parameter @ other
            print(result)
            expected = torch.tensor(expected, dtype=torch.float)

            print(f"parameter.value:\n{parameter.value}")
            print(f"other:\n{other}")
            print(f"got:\n{result}")
            print(f"expected:\n{expected}")

            assert torch.all(result == expected)

            # Confirm that gradients are being computed for parameter.value

            # After the set(value) no gradient should exist:
            assert parameter.value.grad is None

            # Compute a scalar function of `result` and propagate gradients backward:
            torch.mean(result).backward()

            # Now the gradient should exist.
            assert parameter.value.grad is not None
