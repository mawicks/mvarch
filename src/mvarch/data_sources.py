import logging

from typing import Callable, Dict, Iterable, Union
import zipfile

# Third party modules
import numpy as np
import pandas as pd
import yfinance as yf

# Local modules
from deep_volatility_models import util

logging.basicConfig(level=logging.INFO)

DataSource = Callable[[Union[str, Iterable[str]]], Dict[str, pd.DataFrame]]


def YFinanceSource() -> DataSource:
    """
    Sample usage:
    >>> from deep_volatility_models import data_sources
    >>> symbols = ["SPY", "QQQ"]
    >>> ds = data_sources.YFinanceSource()
    >>> response = ds(symbols)
    >>> response["SPY"][:4][['open', 'close']]  # doctest: +NORMALIZE_WHITESPACE
                    open     close
    date
    1993-02-01  43.96875  44.25000
    1993-02-02  44.21875  44.34375
    1993-02-03  44.40625  44.81250
    1993-02-04  44.96875  45.00000
        >>>
    """

    def _add_columns(df):
        new_df = df.dropna().reset_index()
        rename_dict = {c: util.rename_column(c) for c in new_df.columns}
        log_return = np.log(new_df["Adj Close"] / new_df["Adj Close"].shift(1))
        new_df = new_df.assign(log_return=log_return)
        new_df.rename(columns=rename_dict, inplace=True)
        new_df.set_index("date", inplace=True)
        return new_df

    def price_history(symbol_set: Union[Iterable[str], str]) -> Dict[str, pd.DataFrame]:

        # Convert symbol_set to a list
        symbols = util.to_symbol_list(symbol_set)

        # Do the download
        df = yf.download(
            symbols, period="max", group_by="ticker", actions=True, progress=False
        )
        response = {}

        for symbol in symbols:
            # The `group_by` option for yf.download() behaves differently when there's only one symbol.
            # Always return a dictionary of dataframes, even for one symbol.
            if len(symbols) > 1:
                symbol_df = df[symbol]
            else:
                symbol_df = df

            response[symbol] = (
                _add_columns(symbol_df).dropna().applymap(lambda x: round(x, 6))
            )

        return response

    return price_history


def HugeStockMarketDatasetSource(zip_filename) -> DataSource:
    """
    Sample usage
    >>> from deep_volatility_models import data_sources
    >>> symbols = ["SPY", "QQQ"]
    >>> ds = data_sources.HugeStockMarketDatasetSource('archive.zip')
    >>> response = ds(symbols)
    """

    def _add_columns(df):
        new_df = df.dropna().reset_index()
        rename_dict = {c: util.rename_column(c) for c in new_df.columns}
        new_df.rename(columns=rename_dict, inplace=True)

        log_return = np.log(new_df["close"] / new_df["close"].shift(1))
        new_df = new_df.assign(log_return=log_return)

        new_df.set_index("date", inplace=True)
        return new_df

    def price_history(symbol_set: Union[Iterable[str], str]) -> Dict[str, pd.DataFrame]:

        # Convert symbol_set to a list
        symbols = util.to_symbol_list(symbol_set)
        response = {}

        with zipfile.ZipFile(zip_filename, "r") as open_zipfile:
            for symbol in symbols:
                found = False
                for prefix in ["Data/Stocks", "Data/ETFs"]:
                    try:
                        name = f"{prefix}/{symbol.lower()}.us.txt"
                        symbol_df = pd.read_csv(open_zipfile.open(name))
                        response[symbol] = (
                            _add_columns(symbol_df)
                            .dropna()
                            .applymap(lambda x: round(x, 6))
                        )
                        found = True
                    except KeyError:
                        pass
                if not found:
                    raise ValueError(
                        f"Symbol {symbol} not found in Huge Stock Market Dataset"
                    )

        return response

    return price_history


if __name__ == "__main__":  # pragma: no cover
    symbols = ["spy", "qqq"]
    ds = YFinanceSource()
    response = ds(symbols)

    for k, v in response.items():
        print(f"{k}:\n{v.head(3)}")
