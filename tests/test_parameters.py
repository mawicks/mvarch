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
    "parameter_type, value, expect_value_error, other, expected",
    [
        #
        # ScalarParameter test cases
        (ScalarParameter, 2.0, False, [[1, 2], [3, 9]], [[2, 4], [6, 18]]),
        (ScalarParameter, [1, 2], True, None, None),
        (ScalarParameter, [[1, 2], [3, 4]], True, None, None),
        #
        # DiagonalParameter test cases
        (DiagonalParameter, [5, 7], False, [[1, 2], [3, 9]], [[5, 10], [21, 63]]),
        (DiagonalParameter, 2.0, True, None, None),
        (DiagonalParameter, [[1, 2], [3, 4]], True, None, None),
        #
        # TriangularParameter test cases
        (
            TriangularParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            [[5, 10], [40, 113]],
        ),
        (TriangularParameter, 2.0, True, None, None),
        (TriangularParameter, [1, 2], True, None, None),
        (TriangularParameter, [[1, 2], [3, 4]], True, None, None),  # Not triangular
        #
        # FullParameter test cases
        (
            FullParameter,
            [[5, 0], [7, 11]],
            False,
            [[1, 2], [3, 9]],
            [[5, 10], [40, 113]],
        ),
        (
            FullParameter,
            [[5, 7], [11, 13]],
            False,
            [[1, 2], [3, 9]],
            [
                [26, 73],
                [50, 139],
            ],
        ),
        (FullParameter, 2.0, True, None, None),
        (FullParameter, [1, 2], True, None, None),
    ],
)
def test_scalar_parameter(parameter_type, value, expect_value_error, other, expected):
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
        expected = torch.tensor(expected, dtype=torch.float)

        result = parameter @ other

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
