import datetime as dt
import logging
from typing import Union

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """DataSet subclass for time series"""

    def __init__(self, series, sequence_length, stride=1):
        if stride <= 0:
            raise ValueError()

        self._series = np.array(
            series
        )  # Use np.array instead of tuple/list to eliminate performance warning.

        self._sequence_length = sequence_length
        self._stride = stride
        self._length = (len(self._series) - sequence_length) // stride + 1

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
            if len(result.shape) == 1:
                return result
            else:
                return result.t()
        else:
            raise IndexError()


class TargetSelection(torch.utils.data.Dataset):
    """Split time series slices into covariates and target"""

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
        if len(t.shape) == 1:
            if self._target_dim > 0:
                covariates = t[: -self._target_dim]
                target = t[-self._target_dim :]
            else:
                covariates = t[:]
                target = None
        else:
            if self._target_dim > 0:
                covariates = t[:, : -self._target_dim]
                target = t[:, -self._target_dim :]
            else:
                covariates = t[:, :]
                target = None

        result = {"covariates": covariates}
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
