# Standard Python
import datetime as dt
import logging
from typing import Callable, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd  # type: ignore

import numpy as np
import pickle
import torch

REFRESH = True


# Local modules
from .data_sources import HugeStockMarketDatasetSource, YFinanceSource
from .stock_data import (
    PriceHistoryConcatenator,
    FileSystemStore,
    CachingSymbolHistoryLoader,
)

from .model_factory import model_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)

if torch.cuda.is_available():
    dev = "cuda:0"
# elif torch.has_mps:
#    dev = "mps"
else:
    dev = "cpu"

device = torch.device(dev)


def prepare_data(
    history_loader: Callable[
        [Union[str, Iterable[str]]], Iterator[Tuple[str, pd.DataFrame]]
    ],
    symbol_list: Iterable[str],
    start_date: Union[dt.date, None] = None,
    end_date: Union[dt.date, None] = None,
):
    combiner = PriceHistoryConcatenator()

    # Refresh historical data
    logging.info("Reading historical data")
    full_history = combiner(history_loader(symbol_list))
    inference_data = full_history.loc[
        start_date:end_date, (symbol_list, "log_return")  # type: ignore
    ]
    logging.info(
        "Inference data ranges from "
        f"{inference_data.index[0].date()} to {inference_data.index[-1].date()}"
    )
    return inference_data


def check_for_duplicates(symbol_list):
    d = {}
    for s in symbol_list:
        d[s] = d.get(s, 0) + 1
    duplicates = [s for s in d.keys() if d[s] > 1]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate symbols: {duplicates}")


def run(
    model_file,
    periods,
    samples,
    use_hsmd=None,
    start_date=None,
    end_date=None,
    output_file=None,
    device=device,
    pad=True,
):
    # Rewrite symbols with deduped, uppercase versions
    model = torch.load(model_file)
    symbols = list(map(str.upper, model["symbols"]))

    # Check for duplicates
    check_for_duplicates(symbols)
    logging.debug(f"periods: {periods}")
    logging.debug(f"samples: {samples}")
    logging.debug(f"device: {device}")
    logging.debug(f"symbols: {symbols}")
    logging.debug(f"Start date: {start_date}")
    logging.debug(f"End date: {end_date}")
    logging.debug(f"Device: {device}")

    data_store = FileSystemStore("inference_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, REFRESH)

    inference_data = prepare_data(
        history_loader,
        symbols,
        start_date=start_date,
        end_date=end_date,
    )

    logging.debug(f"inference_data:\n {inference_data}")

    inference_observations = torch.tensor(
        inference_data.values, dtype=torch.float, device=device
    )
    logging.debug(f"inference observations:\n {inference_observations}")

    result = model["model"].simulate(inference_observations, periods, samples)
    log_returns = result[0]

    # Padding makes the "day 0" return be zero so that "day 0"
    # represents the most recent closing price and "day 1" is one day
    # in the future.  Without padding, the 0th elment is the first
    # prediction which is one day in the future.

    if pad:
        zeros = torch.zeros(samples, 1, len(symbols), device=device)
        log_returns = torch.cat((zeros, log_returns), dim=1)

    total_returns = torch.exp(torch.cumsum(log_returns, dim=1))

    logging.debug(f"Log returns shape: {log_returns.shape}")
    logging.debug(f"log returns: \n{log_returns}")
    logging.debug(f"total returns: \n{total_returns}")

    if output_file:
        # For compatibility with another tool, the simulation must be dimmensioned
        # as (batch_size, symbols, periods, simulations) - batch should be 1.
        # But the above code leaves it in exactly reversed order (simulations, periods, symbols)
        # Add a batch dimension:
        total_returns = total_returns.unsqueeze(3)

        result = {
            "date": dt.datetime.today(),
            "return_simulations": torch.permute(total_returns, (3, 2, 1, 0)),
        }

        pickle.dump(result, output_file)


@click.command()
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option(
    "--periods",
    "-p",
    default=5,
    show_default=True,
    type=int,
    help="Number of periods (days) to simulate.",
)
@click.option(
    "--samples",
    "-s",
    default=1,
    show_default=True,
    type=int,
    help="Number of simulated outcomes to sample",
)
@click.option(
    "--start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for state",
)
@click.option(
    "--end-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Final date of data used for state",
)
@click.option(
    "--output_file",
    type=click.File("wb"),
    help="Output file for simulation results (pickle file)",
)
@click.argument(
    "model_file",
    type=click.File("rb"),
)
def main_cli(
    use_hsmd,
    periods,
    samples,
    start_date,
    end_date,
    output_file,
    model_file,
):

    if start_date:
        start_date = start_date.date()

    if end_date:
        end_date = end_date.date()

    run(
        model_file=model_file,
        periods=periods,
        samples=samples,
        use_hsmd=use_hsmd,
        start_date=start_date,
        end_date=end_date,
        output_file=output_file,
        device=device,
    )


if __name__ == "__main__":
    main_cli()
