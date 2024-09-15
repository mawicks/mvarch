import datetime as dt
import logging
import random
from typing import Union

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import torch
import torch.utils.data

from mvarch.deep_learning.transformations import (
    proper_get_portfolio_returns,
    random_allocation,
)


class Dataset(torch.utils.data.Dataset):
    """DataSet subclass for time series.
    If the time series is multi-dimensional, this Dataset assumes that
    time is the first dimension.  This is consistent with the dimension
    order used in RNNS, and transformer models, but *not* the convention
    best suited for 1-d convolutional networks where the time dimension
    comes last.  Because we're breaking time up into windows,  it's
    easier if time is the first dimension and we can ignore any other
    dimensions.
    """

    def __init__(self, series, sequence_length, stride=1):
        if stride <= 0:
            raise ValueError()

        self._series = np.array(
            series
        )  # Use np.array instead of tuple/list to eliminate performance warning.

        self._sequence_length = sequence_length
        self._stride = stride

        self._length = (self._series.shape[0] - sequence_length) // stride + 1

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0:
            index = self._length + index

        if index >= 0 and index < self._length:
            start = index * self._stride
            result = torch.tensor(
                self._series[start : start + self._sequence_length], dtype=torch.float
            )
            return result
        else:
            raise IndexError()


class TargetSelection(torch.utils.data.Dataset):
    """Split time series slices into window and target

    The order of the dimensions follows the convention described above
    in the Dataset class of this module.
    """

    def __init__(self, time_series_dataset, target_dim=1, encoded_symbol=None):
        """Generally, the stride used to construct Dataset should be equal to
        target_dim

        """
        self._time_series_dataset = time_series_dataset
        self._target_dim = target_dim
        if encoded_symbol is not None:
            self._symbol = torch.tensor(encoded_symbol)
        else:
            self._symbol = None

    def __len__(self):
        return len(self._time_series_dataset)

    def __getitem__(self, index):
        t = self._time_series_dataset[index]
        if self._target_dim > 0:
            window = t[: -self._target_dim]
            target = t[-self._target_dim :]
        else:
            window = t[:]
            target = None

        result = {"window": window}
        if self._target_dim > 0:
            result["target"] = target
        if self._symbol is not None:
            result["encoded_symbol"] = self._symbol

        return result


class MultiSymbolDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        context_size=128,
        target_dim=1,
        start_date: Union[dt.date, None] = None,
        end_date: Union[dt.date, None] = None,
        encoder: Union[dict[str, int], None] = None,
        decoder: Union[list[str], None] = None,
    ):
        if encoder is None:
            self._encoder = {}
        else:
            self._encoder = encoder

        if decoder is None:
            self._decoder = []
        else:
            self._decoder = decoder

        datasets = []
        for symbol, symbol_history in data.items():
            if symbol not in self._encoder:
                self._encoder[symbol] = len(self._encoder)
                self._decoder.append(symbol)

            encoded_symbol = self._encoder[symbol]

            dataset = symbol_history.loc[start_date:end_date, "log_return"]
            if len(dataset) > 0:
                datasets.append(
                    TargetSelection(
                        Dataset(
                            dataset,
                            context_size + target_dim,
                        ),
                        target_dim,
                        encoded_symbol=encoded_symbol,
                    )
                )
            else:
                logger.warning(f"Symbol {symbol} has no data")

        self._dataset = torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]

    def encoder(self):
        return self._encoder

    def decoder(self):
        return self._decoder

    def symbol_count(self) -> int:
        return len(self._encoder)


class PortfolioDataset(torch.utils.data.Dataset):
    """
    This is a dataset where each item is a portfolio of time series.
    The portfolio is a subset of all symbols in the dataset of size
    `portfolio_dim`.

    The index is ordered so that the first time index and the first
    portfolio correspond to index 0.  The first time index and the
    second portfolio correspond to index 1, and so on until all
    of the portfolios at the lowest time index have been returned.

    Initially, the portfolios are composed from symbols in order, i.e.,
    the portfolio 0 consists of symbols[0:portfolio_dim], etc.  The
    portfolios can be any partition of symbols and can be set
    differently for each time step.  Intially they are all the same.

    This dataset will likely be used with a dataloader that allows the
    index to be shuffled.  Shuffling the index on each epoch won't
    randomize the composition of the portfolios for that purpose.  For
    that purpose, a randomize_portfolios() call changes the composition
    of the portfolios.

    Betwen calls to randomize_portfolios() the values returned by
    __getitem__() are deterministic.
    """

    def __init__(self, data, window_dim, portfolio_dim, target_dim, symbols: list[str]):
        dataset = Dataset(data, window_dim + target_dim, stride=1)
        self._dataset = TargetSelection(dataset, target_dim)
        self._symbols = symbols
        self._portfolio_dim = portfolio_dim
        if len(symbols) % portfolio_dim != 0:
            raise ValueError(
                f"Number of symbols {len(symbols)} must be divisible by portfolio dimension ({portfolio_dim})"
            )
        self._num_portfolios = len(symbols) // portfolio_dim
        self._portfolios = (
            torch.tensor(range(len(symbols)))
            .reshape(self._num_portfolios, -1)
            .unsqueeze(0)
            .expand(len(dataset), self._num_portfolios, self._portfolio_dim)
        )
        self._encoded_symbol = np.array(range(len(symbols)))

    def randomize_portfolios(self):
        """
        Call this when you want to change the assignment of symbols to
        portfolios.  This might typically be called at the beginning of
        each training epoch.  Shuffling the indexes in the call to
        __item__(), which would be performed by the dataloaders, will
        not shuffle the assignment of symbols to portfolios.  This
        method is necessary to do that.

        Betwen calls to randomize_portfolios(), the values returned by
        __getitem__() are deterministic

        """
        portfolios = [
            list(range(len(self._symbols))) for _ in range(len(self._dataset))
        ]
        for p in portfolios:
            random.shuffle(p)

        self._portfolios = torch.tensor(portfolios).reshape(
            len(self._dataset), self._num_portfolios, self._portfolio_dim
        )

    def __len__(self):
        return len(self._dataset) * self._num_portfolios

    def __getitem__(self, index):
        time_series_index = index // self._num_portfolios
        portfolio_index = index % self._num_portfolios

        time_series = self._dataset[time_series_index]
        window = time_series["window"]
        target = time_series["target"]

        portfolio = self._portfolios[time_series_index][portfolio_index]
        portfolio_window = window[:, portfolio]
        portfolio_target = target[:, portfolio]
        portfolio_encoding = portfolio

        return {
            "window": portfolio_window,
            "target": portfolio_target,
            "encoded_symbol": portfolio_encoding,
        }


class RandomPortfolioDataset(torch.utils.data.Dataset):
    """
    This is a wrapper around PortfolioDataset that generates the time
    series that would be achieved by choosing a random allocation of
    assets in that portfolio.  The randomization occurs on each call to
    __getitem__().
    """

    def __init__(self, portfolio_dataset: PortfolioDataset):
        self._portfolio_dataset = portfolio_dataset

    def __getitem__(self, index):
        portfolio = self._portfolio_dataset[index]
        window = portfolio["window"]
        target = portfolio["target"]

        allocation = random_allocation(1, window.shape[1])
        aggregate_window = proper_get_portfolio_returns(
            allocation, window, reverse=True
        ).squeeze(0)
        aggregate_target = proper_get_portfolio_returns(
            allocation, target, reverse=False
        ).squeeze(0)
        return {
            "window": aggregate_window,
            "target": aggregate_target,
            "encoded_symbol": None,
        }

    def randomize_portfolios(self):
        """Call randomize_portfolios on the underlying PortfolioDataset"""
        self._portfolio_dataset.randomize_portfolios()
