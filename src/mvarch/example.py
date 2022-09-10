import mvarch
import numpy as np
import yfinance as yf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

symbols = ("SPY", "QQQ", "BND", "VNQ")

data = yf.download(symbols)

# First dropna() drops rows without data for all symbols
# Second dropna() drops first day of price history which has no return.
# log1p on the whole thing produces the log returns.
df = np.log1p(data.loc[:, ("Adj Close", symbols)].dropna().pct_change().dropna())
fit_history = df.values
model = mvarch.model_factory(distribution="studentt", mean="zero", constraint="none")

print("starting fit()...")
model.tune_all = True
model.fit(fit_history)
print(f"Likelihood: {model.mean_log_likelihood(fit_history):.4f}")

# Truncaate the data for summary purposes:
evaluate_tail = df.index[-500:]
evaluate_history = df.loc[evaluate_tail].values

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
std_dev = np.std(simulated_returns, axis=0)[SIMULATION_PERIODS - 1]
# Calculate the correlation coefficiens from the simulation:
corr_coef = np.corrcoef(simulated_returns[:, SIMULATION_PERIODS - 1, :], rowvar=False)
print(
    f"Std dev of simulation period ({SIMULATION_PERIODS} days) total returns:\n{std_dev}"
)
print(
    f"Correlation of simulation period ({SIMULATION_PERIODS} days) total returns:\n{corr_coef}"
)

# This data is plotted below

plt.style.use("ggplot")

fig1, ax1 = plt.subplots(2, 2)
fig1.set_figwidth(16)
fig1.set_figheight(5)
for i, symbol in enumerate(symbols):
    ax1[i // 2, i % 2].set_title(
        f"{symbol} - Annualized Daily Return and Annualized Volatilty"
    )
    ax1[i // 2, i % 2].plot(
        evaluate_tail,
        np.sqrt(252) * evaluate_history[:, i],
        label="Annualized Daily Return",
    )
    ax1[i // 2, i % 2].plot(
        evaluate_tail,
        np.sqrt(252) * model.distribution.std_dev() * uv_scale_history[:, i],
        label="Annualized Volatility Estimate",
    )
    ax1[i // 2, i % 2].legend()
fig1.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
fig1.savefig("fig1.png")
fig1.show()


last_adj_close = data.loc[data.index[-1], ("Adj Close", symbols)].values
simulated_adj_close = last_adj_close * simulated_returns

fig2, ax2 = plt.subplots(2, 2)
fig2.set_figwidth(16)
fig2.set_figheight(5)
for i, symbol in enumerate(symbols):
    ax2[i // 2, i % 2].set_title(
        f"{symbol} - Adj Close (Past Actual Values and Future Sampled Values)"
    )
    ax2[i // 2, i % 2].plot(
        range(len(evaluate_tail)),
        data.loc[evaluate_tail, ("Adj Close", symbol)],
        label=f"{symbol} history",
    )
    ax2[i // 2, i % 2].plot(
        range(len(evaluate_tail), len(evaluate_tail) + SIMULATION_PERIODS),
        simulated_adj_close[0, :, i],
        label=f"{symbol} sample",
    )
    ax2[i // 2, i % 2].plot(
        (len(evaluate_tail) - 0.5) * np.array([1, 1]),
        ax2[i // 2, i % 2].get_ylim(),
        "k-.",
    )
    ax2[i // 2, i % 2].legend()
fig2.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
fig2.savefig("fig2.png")
fig2.show()
