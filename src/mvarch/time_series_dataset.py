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
            covariates = t[: -self._target_dim]
            target = t[-self._target_dim :]
        else:
            covariates = t[:, : -self._target_dim]
            target = t[:, -self._target_dim :]

        if self._symbol is None:
            return {"covariates": covariates, "target": target}
        else:
            return {
                "encoded_symbol": self._symbol,
                "covariates": covariates,
                "target": target,
            }


class MultiSymbolDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, pd.DataFrame], context_size=128):
        self._encoder = {}
        self._decoder = []

        datasets = []
        for symbol_encoding, (symbol, symbol_history) in enumerate(data.items()):
            datasets.append(
                TargetSelection(
                    Dataset(symbol_history["log_return"], context_size + 1),
                    encoded_symbol=symbol_encoding,
                )
            )
            if symbol in self._encoder:
                raise ValueError(f"{symbol} is duplicated")

            self._encoder[symbol] = symbol_encoding
            self._decoder.append(symbol)
        self._dataset = torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]

    def encoder(self):
        return self._encoder

    def decoder(self):
        return self._decoder
