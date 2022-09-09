import mvarch
import numpy as np
import yfinance as yf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

import logging

logging.basicConfig(level=logging.INFO)

DISTRIBUTION = "studentt"
SIMULATION_PERIODS = 128

# symbols = ("NFLX", "SPY", "TWTR", "XOM")
symbols = ("SPY", "QQQ", "BND", "VNQ")

data = yf.download(symbols)
# First dropna() drops rows without data for all symbols
# Second dropna() drops first day of price history which has no return.
# log1p on the whole thing produces the log returns.
df = np.log1p(data.loc[:, ("Adj Close", symbols)].dropna().pct_change().dropna())
fit_history = df.loc[df.index[:]].values
plot_tail = df.index[-500:]
plot_history = df.loc[plot_tail].values

model = mvarch.model_factory(distribution=DISTRIBUTION, mean="zero", constraint="none")

print("starting fit()...")
model.tune_all = True
model.fit(fit_history)
print(f"Likelihood: {model.mean_log_likelihood(fit_history):.4f}")

mv_scale, uv_scale, mean = model.predict(plot_history)[3:]
plt.style.use("ggplot")

fig1, ax1 = plt.subplots(2, 2)
fig1.set_figwidth(16)
fig1.set_figheight(5)
for i, symbol in enumerate(symbols):
    ax1[i // 2, i % 2].set_title(
        f"{symbol} - Annualized Daily Return and Annualized Volatilty"
    )
    ax1[i // 2, i % 2].plot(
        plot_tail, np.sqrt(252) * plot_history[:, i], label="Annualized Daily Return"
    )
    ax1[i // 2, i % 2].plot(
        plot_tail,
        np.sqrt(252) * model.distribution.std_dev() * uv_scale[:, i],
        label="Annualized Volatility Estimate",
    )
    ax1[i // 2, i % 2].legend()
fig1.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
fig1.savefig("fig1.png")
fig1.show()


simulated, mv_scale, uv_scale, mean = model.simulate(
    plot_history, SIMULATION_PERIODS, 1000
)
simulated_returns = np.exp(np.cumsum(simulated.numpy(), axis=1))
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
        range(len(plot_tail)),
        data.loc[plot_tail, ("Adj Close", symbol)],
        label=f"{symbol} history",
    )
    ax2[i // 2, i % 2].plot(
        range(len(plot_tail), len(plot_tail) + SIMULATION_PERIODS),
        simulated_adj_close[0, :, i],
        label=f"{symbol} sample",
    )
    ax2[i // 2, i % 2].plot(
        (len(plot_tail) - 0.5) * np.array([1, 1]), ax2[i // 2, i % 2].get_ylim(), "k-."
    )
    ax2[i // 2, i % 2].legend()
fig2.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
fig2.savefig("fig2.png")
fig2.show()
