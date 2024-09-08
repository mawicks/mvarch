# Standard Python
import datetime as dt
import logging
import math
from typing import Callable, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd  # type: ignore

import torch
import torch.utils
import torch.utils.data
from lightning import LightningModule  # , LightningDataModule
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.trainer.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Local modules
from mvarch.data_sources import HugeStockMarketDatasetSource, YFinanceSource
from mvarch.stock_data import (
    FileSystemStore,
    CachingSymbolHistoryLoader,
)
from mvarch.time_series_dataset import MultiSymbolDataset

import mvarch.deep_learning.models as models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)


def prepare_data(
    history_loader: Callable[
        [Union[str, Iterable[str]]], Iterator[Tuple[str, pd.DataFrame]]
    ],
    symbol_list: Iterable[str],
    encoder: dict[str, int],
    decoder: list[str],
):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=192)
    logging.info("Reading historical data")
    full_history = dict(history_loader(symbol_list))
    logging.info("Done")
    return MultiSymbolDataset(
        full_history,
        context_size=128,
        target_dim=0,
        start_date=start_date,
        end_date=end_date,
        encoder=encoder,
        decoder=decoder,
    )


def run(
    use_hsmd,
    symbols,
    refresh,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, symbols))
    symbols = ["SPY", "QQQ", "EDV", "TYD", "BND", "GLD", "NVDA"]
    symbols = [
        "AAL",
        "AAPL",
        "AMZN",
        "BA",
        "BABA",
        "BAC",
        "BND",
        "DIS",
        "DG",
        "EDV",
        "F",
        "FXG",
        "GLD",
        "GM",
        "GME",
        "IYF",
        "IYR",
        "KO",
        "KR",
        "NVDA",
        "NFLX",
        "NKE",
        "PG",
        "QLD",
        "QQQ",
        "SBUX",
        "UGE",
        "UPS",
        "V",
        "SPY",
        "TYD",
        "XLV",
        "XLY",
        "XMVM",
        "XOM",
    ]

    logging.debug(f"symbols: {symbols}")
    logging.debug(f"refresh: {refresh}")

    loaded_object = torch.load("model.pkl")
    encoder = loaded_object["encoder"]
    # for s in symbols:
    #     if s not in encoder:
    #         raise ValueError(f"{s} was not in training data.")

    model = loaded_object["model"]
    model.eval()

    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, refresh)

    prediction_data = prepare_data(
        history_loader, symbols, loaded_object["encoder"], loaded_object["decoder"]
    )

    logging.debug(f"length of training_data: {len(prediction_data)}")
    logging.debug(f"length of evaluation_data: {len(prediction_data)}")

    loader = torch.utils.data.DataLoader(
        prediction_data, batch_size=len(prediction_data)
    )

    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)

            mus = predictions[:, 0] * 250.0
            sigmas = predictions[:, 1] * math.sqrt(250)
            ratios = predictions[:, 0] / predictions[:, 1]
            decoded_symbols = [
                loaded_object["decoder"][es] for es in batch["encoded_symbol"]
            ]

            for decoded_symbol, mu, sigma, ratio in zip(
                decoded_symbols, mus, sigmas, ratios
            ):
                print(
                    f"{decoded_symbol} mu:{mu:.4f} sigma:{sigma:.4f} ratio:{ratio:.4f}"
                )


@click.command()
@click.option(
    "--symbol",
    "-s",
    multiple=True,
    help="Symbol(s) to include in model (may be specified multiple times)",
)
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    show_default=True,
    help="Refresh cached stock data",
)
def main_cli(
    symbol,
    use_hsmd,
    refresh,
):

    run(
        use_hsmd,
        symbols=symbol,
        refresh=refresh,
    )


if __name__ == "__main__":
    main_cli()
