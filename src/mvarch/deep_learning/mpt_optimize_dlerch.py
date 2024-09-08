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
    PriceHistoryConcatenator,
)

from mvarch.time_series_dataset import MultiSymbolDataset

import mvarch.deep_learning.models as models

WINDOW_SIZE = 256
MAX_ITERATIONS = 1_000
LR = 0.25
RUNS = 50
PATIENCE = 50

MODEL = "transformer-all-data.pkl"

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

    # Refresh historical data
    logging.info("Reading historical data")
    combiner = PriceHistoryConcatenator()
    portfolio_full_history = combiner(history_loader(symbol_list))
    portfolio_log_returns = portfolio_full_history.loc[
        :, (tuple(symbol_list), "log_return")
    ]  # type:ignore
    portfolio_log_returns = portfolio_log_returns.droplevel(1, axis=1)
    logging.info(f"Last date: {portfolio_log_returns.index[-1]}")
    logging.info("Done")
    return torch.tensor(portfolio_log_returns.values[-256:], dtype=torch.float32)


def run(
    use_hsmd,
    symbols,
    refresh,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, symbols))
    ira_symbols = [
        "BND",
        "EDV",
        "FXG",
        "FXL",
        "GLD",
        "QQQ",
        "SPY",
        "TYD",
        "VBK",
        "VNQ",
        "XLV",
        "XMVM",
    ]
    taxable_symbols = [
        "BND",
        "EDV",
        "FXG",
        "FXL",
        "QQQ",
        "SPY",
        "TYD",
        "VBK",
        "VNQ",
        "XLV",
        "XMVM",
    ]
    other_symbols = ["BND", "EDV", "QQQ", "SPY", "TYD"]

    test_symbols = [
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

    symbols = ira_symbols

    refresh = False

    logging.debug(f"symbols: {symbols}")
    logging.debug(f"refresh: {refresh}")

    loaded_object = torch.load(MODEL)
    encoder = loaded_object["encoder"]

    model = loaded_object["model"]
    model.eval()

    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, refresh)

    historical_returns = prepare_data(
        history_loader, symbols, loaded_object["encoder"], loaded_object["decoder"]
    )

    batch = {
        "covariates": historical_returns.permute((1, 0)),
        "encoded_symbol": None,
    }
    mus, sigmas = model(batch).unbind(1)
    for symbol, mu, sigma in zip(symbols, mus, sigmas):
        ratio = mu / sigma
        print(f"{symbol:>5s}  mu: {250*mu:6.3f}  ratio: {ratio:6.3f}")

    allocations = []
    for i in range(RUNS):
        allocations.append(get_optimal_allocation(symbols, historical_returns, model))

    mean_allocation = torch.mean(torch.stack(allocations, dim=0), dim=0)

    for i, allocation in enumerate(allocations):
        alloc = [f"{a:0.3f}" for a in allocation]
        print(f"{i}) {alloc}")

    print(f"mean: {mean_allocation}")

    print(f"symbols: {symbols}")


def get_optimal_allocation(symbols, historical_returns, model):
    allocation = torch.randn(len(symbols), dtype=torch.float32)
    # Need to set requires_grad outside of the randn() because the
    # multiplications would make it a non-leaf tensor.
    allocation.requires_grad = True
    optim = torch.optim.Adam(params=[allocation], lr=LR, maximize=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=20
    )
    best_perf = -float("inf")
    best_mu = best_sigma = None

    for iteration, _ in enumerate(range(MAX_ITERATIONS)):
        optim.zero_grad()

        portfolio_returns = proper_get_portfolio_returns(historical_returns, allocation)

        batch = {
            "covariates": portfolio_returns.unsqueeze(0),
            "encoded_symbol": None,
        }
        mu, sigma = model(batch)[0]
        performance = 250 * (mu - 0.25 * sigma)

        if float(performance) > best_perf:
            best_perf = float(performance)
            best_iteration = iteration
            best_allocation = torch.softmax(allocation.detach(), 0)
            best_mu = float(mu)
            best_sigma = float(sigma)

        if iteration - best_iteration >= PATIENCE:
            break

        scheduler.step(performance)
        performance.backward()
        optim.step()

        if iteration % 10 == 0:
            print(
                f"{iteration}) lr:{scheduler.get_last_lr()} mu:{250.0*float(mu):.3f} sigma:{250.0*float(sigma):.3f} ratio:{float(mu/sigma):.3f} perf:{float(performance):0.4f}"
            )

    print(f"best perf: {best_perf:.4f} at iter {best_iteration}")
    print(f"best allocation: {best_allocation}")
    return best_allocation


def simple_get_portfolio_returns(historical_returns, allocation):
    """Use a simple linear formula to approximate the historical log-returns
    of a portfolio, given the historical log-returns of each asset in
    the portfolio."""

    return torch.sum(historical_returns * torch.softmax(allocation, 0), dim=1)


def proper_get_portfolio_returns(historical_returns, allocation):
    """Use a more complex formula to get the historical log-returns of a portfolio,
    given the historical log-returns of each asset int the portfolio.
    Assume the portfolio allocation is chosen today, then back-propogate
    the returns of each asset in the portfolio to get the historical
    asset values. Add then together in the value space rather then in
    the log-return space.
    """
    log_allocation = torch.log_softmax(allocation, 0)
    # Compute the portfolio value history, assuming allocation
    # represents today's value allocation.
    portfolio_value_history = torch.flip(
        torch.sum(
            torch.exp(
                log_allocation - torch.cumsum(torch.flip(historical_returns, [0]), 0)
            ),
            1,
        ),
        [0],
    )
    # Compute the log returns by computing the log of the value history
    # and diff-ing.  Append a 1 because the total value today is 1.  The
    # diff drops a dimension and this prevents losing the information in
    # the most recent day's return.
    x = torch.diff(
        torch.concat(
            [
                torch.log(portfolio_value_history),
                torch.zeros(1),
            ]
        )
    )
    return x


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
