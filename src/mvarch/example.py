import yfinance as yf  # type: ignore
import numpy as np

import mvarch

symbols = ("BND", "GLD", "SPY", "QQQ")
data = yf.download(symbols)
# First dropna() drops rows without data for all symbols
# Second dropna() drops first day of price history which has no return.
# log1p on the whole thing produces the log returns.
df = np.log1p(data.loc[:, ("Adj Close", symbols)].dropna().pct_change().dropna())

model = mvarch.model_factory(distribution="studentt", mean="arma")

print("starting fit()...")
model.fit(df.values)
print("fit() finished")

model.simulate(1000, 20, df.values)
