from pathlib import Path
from typing import Callable, Any, Iterable

import pytest

import pandas as pd
import torch

from mvarch.time_series_dataset import Dataset, TargetSelection, MultiSymbolDataset
from mvarch import stock_data


def test_dataset():
    d = Dataset(range(10), 3)
    assert len(d) == 8
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(1, 4))
    assert tuple(d[-1]) == tuple(range(7, 10))
    assert tuple(d[-2]) == tuple(range(6, 9))
    assert tuple(d[-8]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[8]
    with pytest.raises(IndexError):
        d[-9]

    d = Dataset(range(10), 3, stride=2)
    assert len(d) == 4
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(2, 5))
    assert tuple(d[-1]) == tuple(range(6, 9))
    assert tuple(d[-2]) == tuple(range(4, 7))
    assert tuple(d[-4]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[4]
    with pytest.raises(IndexError):
        d[-5]

    with pytest.raises(ValueError):
        Dataset(range(10), 3, stride=0)


def test_target_selection():
    d = Dataset(range(10), 3, stride=2)
    t = TargetSelection(d, target_dim=1)

    i = iter(t)
    d = next(i)
    assert d["target"] == 2
    d = next(i)
    assert d["target"] == 4
    d = next(i)
    assert d["target"] == 6
    d = next(i)
    assert d["target"] == 8

    with pytest.raises(StopIteration):
        next(i)


CONTEXT_SIZE = 6


def multisymbol_dataset(data_source, tmp_path):
    symbols = ["ABC", "DEF"]

    tmp_path_store = stock_data.FileSystemStore(str(tmp_path))
    caching_download = stock_data.CachingDownloader(
        data_source,
        tmp_path_store,
        stock_data.SymbolHistoryWriter,
        overwrite_existing=False,
    )

    response = caching_download(symbols)
    dataset = MultiSymbolDataset(response, context_size=CONTEXT_SIZE)

    return symbols, response, dataset


def test_multisymbol_dataset_window(
    data_source: Callable[..., dict[Any, pd.DataFrame]], tmp_path: Path
):
    symbols, response, dataset = multisymbol_dataset(data_source, tmp_path)

    time_series_lengths = [len(response[symbol]) for symbol in symbols]
    training_record_counts = [l - CONTEXT_SIZE for l in time_series_lengths]

    # Confirm expected number of training records
    assert len(dataset) == sum(training_record_counts)

    # These are not very extensive checks...
    # Mock data_source has strictly increasing log return sequence,
    # so every window is a strictly increasing series and
    # every target should be larger than any point in the window
    for datum in dataset:
        assert datum["target"] > torch.max(datum["covariates"])


def test_multisymbol_dataset_encoding(
    data_source: Callable[..., dict[Any, pd.DataFrame]], tmp_path: Path
):
    symbols, __ignored__, dataset = multisymbol_dataset(data_source, tmp_path)

    encoder = dataset.encoder()
    decoder = dataset.decoder()

    half_length = len(dataset) // 2

    # iter() around dataset is necessary to keep type check happy
    # since PyTorch datasets don't implement __iter__, which is
    # technicaly required to be type-compatible with enumerate(), but
    # Python can create an iterator using  __getitem__().

    for number, datum in enumerate(iter(dataset)):
        # Make sure the encoder/decoder round trip works by decoding the
        # encoding, re-encoding it and checking for a match.
        decoded_symbol = decoder[datum["encoded_symbol"]]
        assert encoder[decoded_symbol] == datum["encoded_symbol"]

        # Having the round-trip work, doesn't necessarily mean the
        # symbol was encoded correctly.  Since the mock time series are
        # identical for 'ABC' and 'DEF', we can't directly check that
        # they were encoded correctly.  Instead, we'll make an
        # assumption that MultiSymbolDataset and CachingDownloader
        # implementations both preserve the order of `symbols` and check
        # that the first half of dataset is encoded as 'ABC' and second
        # half is encoded as 'DEF'.

        if number < half_length:
            assert decoded_symbol == symbols[0]
        else:
            assert decoded_symbol == symbols[1]
        number += 1
