# Standard Python
import datetime as dt
import logging
from typing import Callable, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd  # type: ignore

import torch


# Local modules
from .data_sources import HugeStockMarketDatasetSource, YFinanceSource
from .stock_data import (
    PriceHistoryConcatenator,
    FileSystemStore,
    CachingSymbolHistoryLoader,
)

from .model_factory import model_factory

DEFAULT_SEED = 42

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
    # Refresh historical data
    logging.info("Reading historical data")

    combiner = PriceHistoryConcatenator()

    symbol_list = sorted(symbol_list)
    full_history = combiner(history_loader(symbol_list))
    training_data = full_history.loc[
        start_date:end_date, (symbol_list, "log_return")  # type: ignore
    ]
    print(training_data.columns)

    return training_data


def run(
    use_hsmd,
    symbols,
    refresh,
    seed=DEFAULT_SEED,
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
    symbols = list(map(str.upper, set(symbols)))

    logging.info(f"device: {device}")
    logging.info(f"symbols: {symbols}")
    logging.info(f"refresh: {refresh}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Start date: {start_date}")
    logging.info(f"End date: {end_date}")
    logging.info(f"Evaluation/termination start date: {eval_start_date}")
    logging.info(f"Evaluation/termination end date: {eval_end_date}")
    logging.info(f"Mean model: {mean}")
    logging.info(f"Univariate model: {univariate}")
    logging.info(f"Multivariate model: {multivariate}")
    logging.info(f"Distribution: {distribution}")
    logging.info(f"Device: {device}")

    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, refresh)

    torch.random.manual_seed(seed)

    training_data = prepare_data(
        history_loader,
        symbols,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )
    logging.info(f"training_data:\n {training_data}")

    observations = torch.tensor(training_data.values, dtype=torch.float, device=device)
    logging.info(f"observations:\n {observations}")

    model = model_factory(
        distribution=distribution,
        mean=mean,
        univariate=univariate,
        multivariate=multivariate,
        constraint=constraint,
        device=device,
    )

    model.fit(observations)
    # Compute the final loss
    ll = model.mean_log_likelihood(observations)
    logging.info(f"**** Final likelihood (per sample): {ll:.4f} ****")

    result = model.predict(observations)
    if len(result) == 6:
        (
            mv_scale_next,
            uv_scale_next,
            mu_next,
            mv_scale,
            uv_scale,
            mu,
        ) = result
    else:
        (
            uv_scale_next,
            mu_next,
            uv_scale,
            mu,
        ) = result
        mv_scale_next = mv_scale = None
    print("mu_next: \n", mu_next)

    torch.save(multivariate, "model.pt")
    print("mv_scale: ", mv_scale.shape if mv_scale is not None else None)

    # Compute some useful quantities to display and to record
    if mv_scale is not None:
        total_scale = uv_scale.unsqueeze(2).expand(mv_scale.shape) * mv_scale
        covariance = total_scale @ torch.transpose(total_scale, 1, 2)
        total_scale_predicted = (
            uv_scale_next.unsqueeze(1).expand(mv_scale_next.shape) * mv_scale_next
        )
        predicted_covariance = total_scale_predicted @ total_scale_predicted.T

        sigma = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2))
        predicted_sigma = torch.sqrt(torch.diag(predicted_covariance))

        inverse_sigma = torch.diag_embed(sigma ** (-1), dim1=1, dim2=2)
        inverse_predicted_sigma = torch.diag(predicted_sigma ** (-1))

        correlation = inverse_sigma @ covariance @ inverse_sigma
        predicted_correlation = (
            inverse_predicted_sigma @ predicted_covariance @ inverse_predicted_sigma
        )
    else:
        sigma = uv_scale
        predicted_sigma = uv_scale_next
        covariance = predicted_covariance = correlation = predicted_correlation = None

    safe_reference = lambda x: x.numpy if x is not None else None
    result = {
        "transformation": safe_reference(mv_scale),
        "predicted_mv_scale": safe_reference(mv_scale_next),
        "uv_scale": safe_reference(uv_scale),
        "predicted_uv_scale": safe_reference(uv_scale_next),
        "covariance": safe_reference(covariance),
        "predicted_covariance": safe_reference(predicted_covariance),
        "correlation": safe_reference(correlation),
        "predicted_correlation": safe_reference(predicted_correlation),
        "training_data": training_data,
    }

    torch.save(result, "mgarch_output.pt")

    logging.info(
        f"MV transformation estimates:\n{mv_scale if mv_scale is not None else None}"
    )
    logging.info(f"UV scale estiamtes:\n{uv_scale}")
    logging.info(f"mean estiamtes:\n{mu}")
    logging.info(f"correlations:\n{correlation}")


@click.command()
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option("--symbol", "-s", multiple=True, show_default=True)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    show_default=True,
    help="Refresh stock data",
)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
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
    default="none",
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
    use_hsmd,
    symbol,
    refresh,
    seed,
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

    run(
        use_hsmd,
        symbols=symbol,
        refresh=refresh,
        seed=seed,
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
