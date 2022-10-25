import mvarch

import datetime as dt
import numpy as np
import yfinance as yf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

import logging

# Common packages
import click

# Local modules
from mvarch.data_sources import HugeStockMarketDatasetSource, YFinanceSource

from mvarch.stock_data import (
    PriceHistoryConcatenator,
    FileSystemStore,
    CachingSymbolHistoryLoader,
)

ZOOM_START = dt.datetime(year=2009, month=1, day=1)
ZOOM_END = dt.datetime(year=2010, month=10, day=1)

logging.basicConfig(level=logging.INFO)


def run(use_hsmd):
    data_store = FileSystemStore("training_data")
    if use_hsmd:
        data_source = HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = YFinanceSource()

    history_loader = CachingSymbolHistoryLoader(data_source, data_store, False)

    symbols = ("SPY", "QQQ", "BND", "VNQ")

    combiner = PriceHistoryConcatenator()
    full_history = combiner(history_loader(symbols))

    last_close = full_history.loc[full_history.index[-1], (symbols, "close")].values

    # Choose a specific time period for a "zoomed" plot.
    zoom_history = full_history.loc[
        ZOOM_START:ZOOM_END,
        (symbols, "log_return"),
    ]

    fit_history = full_history.loc[:, (symbols, "log_return")].values

    # Truncaate the data for summary purposes:
    TAIL_SIZE = 500
    evaluate_tail = full_history.index[-TAIL_SIZE:]
    evaluate_history = full_history.loc[evaluate_tail, (symbols, "log_return")].values

    model = mvarch.model_factory(
        distribution="studentt", mean="zero", constraint="none"
    )

    print("starting fit()...")
    model.tune_all = True
    model.fit(fit_history)
    print(f"Likelihood: {model.mean_log_likelihood(fit_history):.4f}")

    # 0: Data for "Zoomed" plot
    (
        zoomed_mv_scale_predicted,
        zoomed_uv_scale_predicted,
        zoomed_mean_predicted,
        zoomed_mv_scale_history,
        zoomed_uv_scale_history,
        zoomed_mean_history,
    ) = model.predict(zoom_history.values)

    # Use cases:
    # 1: Get correlation, std deviation, and mean estimates for
    # previous days from history and also for the next business day:
    (
        mv_scale_predicted,
        uv_scale_predicted,
        mean_predicted,
        mv_scale_history,
        uv_scale_history,
        mean_history,
    ) = model.predict(evaluate_history)
    # Predicted next day correlations:
    print(
        f"Next day correlation prediction:\n"
        f"{(mv_scale_predicted @ mv_scale_predicted.T).numpy()}"
    )
    print(
        f"Next day volatility prediction (annualized):\n"
        f"{(np.sqrt(252.) * uv_scale_predicted * model.distribution.std_dev()).numpy()}"
    )

    # This data is plotted below

    # 2: Get simulated results (Monte Carlo simulation) for the next
    # `SIMULATION_PERIODS` days, drawing simulated, drawing from
    # `SIMULATION_SAMPLES` outcomes.

    SIMULATION_PERIODS = 126
    SIMULATION_SAMPLES = 1000

    simulated, mv_scale, uv_scale, mean = model.simulate(
        evaluate_history, SIMULATION_PERIODS, 1000
    )
    simulated_returns = np.exp(np.cumsum(simulated.numpy(), axis=1))

    # Return value has shape (SIMULATION_SAMPLES, SIMULATION_PERIODS, STOCK_SYMBOLS)

    # Calculate the standard deviation of the returns for each variable from the simulation:
    std_dev = np.std(simulated_returns[:, -1, :], axis=0)
    # Calculate the correlation coefficiens from the simulation:
    corr_coef = np.corrcoef(simulated_returns[:, -1, :], rowvar=False)
    print(
        f"Std dev of simulation period ({SIMULATION_PERIODS} days) total returns:\n{std_dev}"
    )
    print(
        f"Correlation of simulation period ({SIMULATION_PERIODS} days) total returns:\n{corr_coef}"
    )

    # This data is plotted below

    plt.style.use("ggplot")
    # **************

    plt.figure(figsize=(6, 5))
    plt.title(f"{symbols[0]} - Annualized Daily Return and Annualized Volatilty")
    plt.plot(
        zoom_history.index,
        np.sqrt(252) * zoom_history.values[:, 0],
        label="Annualized Daily Return",
    )
    plt.plot(
        zoom_history.index,
        np.sqrt(252) * model.distribution.std_dev() * zoomed_uv_scale_history[:, 0],
        label="Annualized Volatility Estimate",
    )
    plt.legend()
    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    plt.savefig("fig0.png")
    plt.show()

    # ***************

    fig1, ax1 = plt.subplots(2, 2)
    fig1.set_figwidth(16)
    fig1.set_figheight(5)
    for i, symbol in enumerate(symbols):
        ax = ax1[i // 2, i % 2]
        ax.set_title(f"{symbol} - Annualized Daily Return and Annualized Volatilty")
        ax.plot(
            evaluate_tail,
            np.sqrt(252) * evaluate_history[:, i],
            label="Annualized Daily Return",
        )
        ax.plot(
            evaluate_tail,
            np.sqrt(252) * model.distribution.std_dev() * uv_scale_history[:, i],
            label="Annualized Volatility Estimate",
        )
        ax.legend()
    fig1.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    fig1.savefig("fig1.png")
    fig1.show()

    print("last_close:\n", last_close)
    print("simulated_returns:\n", simulated_returns)
    simulated_adj_close = last_close * simulated_returns

    fig2, ax2 = plt.subplots(2, 2)
    fig2.set_figwidth(16)
    fig2.set_figheight(5)
    for i, symbol in enumerate(symbols):
        ax = ax2[i // 2, i % 2]
        ax.set_title(
            f"{symbol} - Adj Close  for past {TAIL_SIZE} days and future {SIMULATION_PERIODS} days"
        )
        ax.set_xlabel(
            f"Days - values before {TAIL_SIZE} are history; values after are simulated"
        )
        ax.plot(
            range(len(evaluate_tail)),
            full_history.loc[evaluate_tail, (symbol, "close")],
            label=f"{symbol} history",
        )
        ax.plot(
            range(len(evaluate_tail), len(evaluate_tail) + SIMULATION_PERIODS),
            simulated_adj_close[0, :, i],
            label=f"{symbol} sample",
        )
        ax.plot(
            (len(evaluate_tail) - 0.5) * np.array([1, 1]),
            ax2[i // 2, i % 2].get_ylim(),
            "k-.",
        )
        ax.legend()
    fig2.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    fig2.savefig("fig2.png")
    fig2.show()


@click.command()
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
def main_cli(use_hsmd):
    run(use_hsmd)


if __name__ == "__main__":
    main_cli()
