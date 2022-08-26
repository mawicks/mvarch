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
        (ScalarParameter, 2.0, False, [[1, 2], [3, 9]], False, [[2, 4], [6, 18]]),
        (ScalarParameter, [1, 2], True, None, False, None),
        (ScalarParameter, [[1, 2], [3, 4]], True, None, False, None),
        #
        # DiagonalParameter test cases
        (
            DiagonalParameter,
            [5, 7],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [21, 63]],
        ),
        (DiagonalParameter, [5, 7], False, [[1, 2], [3, 4], [5, 6]], True, None),
        (DiagonalParameter, 2.0, True, None, False, None),
        (DiagonalParameter, [[1, 2], [3, 4]], True, None, False, None),
        #
        # TriangularParameter test cases
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [40, 113]],
        ),
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 4], [5, 6]],
            True,
            None,
        ),
        (TriangularParameter, 2.0, True, None, False, None),
        (TriangularParameter, [1, 2], True, None, False, None),
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
        (
            FullParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            False,
            [[5, 10], [40, 113]],
        ),
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
        (FullParameter, 2.0, True, None, False, None),
        (FullParameter, [1, 2], True, None, False, None),
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
