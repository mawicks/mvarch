from typing import Any, List, Iterable, Optional, Union

import torch


def to_symbol_list(symbols: Union[Iterable[str], str]) -> List[str]:
    """
    This function converts its `symbols` argument to a list of strings.
    It's used as a convenence function so that a caller can provide a
    single symbol to a function rather than in Iterable.
    We also normalize symbols lists by converting the symbols to upper case.

    Arguments:
       symbols: Union[Iterable[str], str]: The collection of symbols to convert or a single symbol
    Returns:
       List[str]: An instantiated list of symbols, hich may the same list passed in
    """

    if isinstance(symbols, str):
        symbols = (symbols,)

    return list(map(str.upper, symbols))


def to_tensor(tensor_like: Any, device: Optional[torch.device] = None):
    if not isinstance(tensor_like, torch.Tensor):
        tensor_like = torch.tensor(tensor_like, dtype=torch.float, device=device)
    return tensor_like


def is_sorted(l: Iterable) -> bool:
    l = tuple(l)
    return all([x <= y for x, y in zip(l, l[1:])])


def rename_column(c: str):
    """
    Standardize column naming.  No spaces and no caps.
    """
    return c.lower().replace(" ", "_")
