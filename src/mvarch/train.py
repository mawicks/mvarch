# Standard Python
import datetime as dt
import logging
from typing import Callable, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd  # type: ignore

import numpy as np
import torch


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
    eval_start_date: Union[dt.date, None] = None,
    eval_end_date: Union[dt.date, None] = None,
):
    combiner = PriceHistoryConcatenator()

    # Refresh historical data
    logging.info("Reading historical data")
    full_history = combiner(history_loader(symbol_list))
    training_data = full_history.loc[
        start_date:end_date, (symbol_list, "log_return")  # type: ignore
    ]
    logging.info(
        "Training data ranges from "
        f"{training_data.index[0].date()} to {training_data.index[-1].date()}"
    )
    evaluation_data = full_history.loc[
        eval_start_date:eval_end_date, (symbol_list, "log_return")  # type: ignore
    ]
    return training_data, evaluation_data


def check_for_duplicates(symbol_list):
    d = {}
    for s in symbol_list:
        d[s] = d.get(s, 0) + 1
    duplicates = [s for s in d.keys() if d[s] > 1]
    if len(duplicates) > 0:
        raise ValueError(f"Duplicate symbols: {duplicates}")


def run(
    use_hsmd,
    symbols,
    refresh,
    output_file=None,
    start_date=None,
    end_date=None,
    eval_start_date=None,
    eval_end_date=None,
    mean="zero",
    univariate="arch",
    multivariate="mvarch",
    constraint="none",
    distribution="normal",
    device=device,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, symbols))

    # Check for duplicates
    check_for_duplicates(symbols)

    logging.debug(f"device: {device}")
    logging.debug(f"symbols: {symbols}")
    logging.debug(f"refresh: {refresh}")
    logging.debug(f"Start date: {start_date}")
    logging.debug(f"End date: {end_date}")
    logging.debug(f"Evaluation/termination start date: {eval_start_date}")
    logging.debug(f"Evaluation/termination end date: {eval_end_date}")
    logging.debug(f"Mean model: {mean}")
    logging.debug(f"Univariate model: {univariate}")
    logging.debug(f"Multivariate model: {multivariate}")
    logging.debug(f"Distribution: {distribution}")
    logging.debug(f"Device: {device}")

    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, refresh)

    training_data, evaluation_data = prepare_data(
        history_loader,
        symbols,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )

    logging.debug(f"training_data:\n {training_data}")
    train_observations = torch.tensor(
        training_data.values, dtype=torch.float, device=device
    )
    logging.debug(f"training observations:\n {train_observations}")

    model = model_factory(
        distribution=distribution,
        mean=mean,
        univariate=univariate,
        multivariate=multivariate,
        constraint=constraint,
        device=device,
    )

    model.fit(train_observations)

    # Compute the final training loss
    train_ll = model.mean_log_likelihood(train_observations)
    logging.info(f"**** Per sample log likelihood (train): {train_ll:.4f} ****")

    if eval_start_date != start_date or eval_end_date != end_date:
        eval_observations = torch.tensor(
            evaluation_data.values, dtype=torch.float, device=device
        )
        logging.info(
            "Evaluation data ranges from "
            f"{evaluation_data.index[0].date()} to {evaluation_data.index[-1].date()}"
        )
        eval_ll = model.mean_log_likelihood(eval_observations)
        logging.info(f"**** Per sample log likelihood (eval): {eval_ll:.4f} ****")
    else:
        eval_ll = train_ll

    result = model.predict(train_observations)
    if len(result) == 6:
        (
            next_mv_scale,
            next_uv_scale,
            next_mean,
        ) = result[:3]
    else:
        next_uv_scale, next_mean = result[:2]
        next_mv_scale = None

    if output_file is not None:
        output = {
            "date": dt.datetime.today(),
            "symbols": symbols,
            "model": model,
            "eval_log_likelihood": round(eval_ll, 4),
            "first_train_date": training_data.index[0].date(),
            "last_train_date": training_data.index[-1].date(),
            "first_eval_date": evaluation_data.index[0].date(),
            "last_eval_date": evaluation_data.index[-1].date(),
            "train_log_likelihood": round(train_ll, 4),
            "next_mv_scale": next_mv_scale,
            "next_uv_scale": next_uv_scale,
            "next_mean": next_mean,
        }

        torch.save(output, output_file)

    if next_mv_scale is not None:
        total_scale = next_uv_scale.unsqueeze(1) * next_mv_scale
        # Next line computes the row norms
        scale = torch.linalg.norm(total_scale, dim=1)
        # Next line normalizes the rows
        t = torch.nn.functional.normalize(total_scale, dim=1)
        corr = t @ t.T
    else:
        scale = next_uv_scale
        corr = None

    volatility = np.sqrt(252.0) * scale * model.distribution.std_dev()
    logging.info(f"Symbols: {symbols}")
    logging.info(
        f"Annualized volatility prediction for next time period: {volatility.numpy()}"
    )
    if corr is not None:
        logging.info(f"Correlation prediction for next time period:\n{corr.numpy()}")


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
@click.option(
    "--output",
    "-o",
    type=click.File("wb"),
    help="Output file for trained model and metadata.",
)
@click.option(
    "--start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for training",
)
@click.option(
    "--end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Final date of data used for training",
)
@click.option(
    "--eval-start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for evaluation/termination",
)
@click.option(
    "--eval-end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last date of data used for evaluation/termination",
)
@click.option(
    "--mean",
    type=click.Choice(["zero", "constant", "arma"], case_sensitive=False),
    default="zero",
    show_default=True,
    help="Type of mean model to use",
)
@click.option(
    "--univariate",
    type=click.Choice(
        ["arch", "none"],
        case_sensitive=False,
    ),
    default="arch",
    show_default=True,
    help="Type of univariate model to use",
)
@click.option(
    "--multivariate",
    type=click.Choice(["mvarch", "none"], case_sensitive=False),
    default="mvarch",
    show_default=True,
    help="Type of multivariate model to use (or 'none')",
)
@click.option(
    "--constraint",
    "-c",
    type=click.Choice(
        ["scalar", "diagonal", "triangular", "none"], case_sensitive=False
    ),
    default="diagonal",
    help="Type of constraint to be applied to multivariate parameters.",
)
@click.option(
    "--distribution",
    "-d",
    type=click.Choice(["normal", "studentt"], case_sensitive=False),
    default="normal",
    help="Error distribution to use.",
)
def main_cli(
    symbol,
    use_hsmd,
    refresh,
    output,
    start_date,
    end_date,
    eval_start_date,
    eval_end_date,
    mean,
    univariate,
    multivariate,
    constraint,
    distribution,
):

    if start_date:
        start_date = start_date.date()

    if end_date:
        end_date = end_date.date()

    if eval_start_date:
        eval_start_date = eval_start_date.date()

    if eval_end_date:
        eval_end_date = eval_end_date.date()

    if univariate == "none" and multivariate == "none":
        print("Univariate model and multivariate model cannot both be 'none'")
        exit(1)

    run(
        use_hsmd,
        symbols=symbol,
        refresh=refresh,
        output_file=output,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
        mean=mean,
        univariate=univariate,
        multivariate=multivariate,
        constraint=constraint,
        distribution=distribution,
        device=device,
    )


if __name__ == "__main__":
    main_cli()
