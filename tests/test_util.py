import pytest

import torch

# Local modules
import mvarch.util as util


@pytest.mark.parametrize(
    "l,expected",
    [
        ([], True),  # Check empty list.
        (range(4), True),  # Make sure iterables work
        (tuple(range(4)), True),  # Make sure instantiated tuples work
        (list(range(4)), True),  # Instantiated lists
        (reversed(range(4)), False),  # Reversed lists should fail.
        ([1, 1, 2, 2], True),  # Check that non-strict inequality is ok.
        ([1] * 5, True),
        ([1, 1, 1, 0], False),  # Edge cases?
    ],
)
def test_is_sorted(l, expected):
    assert util.is_sorted(l) == expected


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        ("foo", ["FOO"]),
        ([], []),
        (["foo"], ["FOO"]),
        (("a", "b"), ["A", "B"]),
        (iter(("x", "y", "z")), ["X", "Y", "Z"]),
    ],
)
def test_to_symbol_list(test_input, expected_output):
    print(f"test input: {test_input}")
    print(f"expected output: {expected_output}")
    assert util.to_symbol_list(test_input) == expected_output


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        ("foo", "foo"),
        ("Foo", "foo"),
        ("a b c", "a_b_c"),
        ("A b C", "a_b_c"),
    ],
)
def test_rename_column(test_input, expected_output):
    print(f"test input: {test_input}")
    print(f"expected output: {expected_output}")
    assert util.rename_column(test_input) == expected_output


@pytest.mark.parametrize(
    "test_input",
    [
        4.0,
        [[1.0, 2.0], [3.0, 4.0]],
        torch.tensor([1.0]),
        torch.tensor([2.0], requires_grad=True),
    ],
)
def test_to_tensor(test_input):
    result = util.to_tensor(test_input, requires_grad=False)
    assert isinstance(result, torch.Tensor)
    assert hasattr(result, "requires_grad") is False or result.requires_grad is False

    result = util.to_tensor(test_input, requires_grad=True)
    assert isinstance(result, torch.Tensor)
    assert result.requires_grad == True
