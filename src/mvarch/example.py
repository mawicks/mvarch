import yfinance as yf  # type: ignore
import numpy as np
import torch

import mvarch

symbols = ("BND", "GLD", "SPY", "QQQ")
data = yf.download(symbols)
# First dropna() drops rows without data for all symbols
# Second dropna() drops first day of price history which has no return.
# log1p on the whole thing produces the log returns.
df = np.log1p(data.loc[:, ("Adj Close", symbols)].dropna().pct_change().dropna())


distribution = mvarch.distributions.StudentTDistribution()
mean_model = mvarch.mean_models.ARMAMeanModel()
univariate_model = mvarch.univariate_models.UnivariateARCHModel(
    mean_model=mean_model, distribution=distribution
)
multivariate_model = mvarch.multivariate_models.MultivariateARCHModel(
    univariate_model=univariate_model
)

print("starting fit()...")
multivariate_model.fit(df.values)
print("fit() finished")

x = multivariate_model.predict(df.values)
next_mv_state, next_uv_state, next_mean_state = x[:3]

output_list = []
mv_scale_list = []
uv_scale_list = []
mean_list = []
for i in range(1000):
    output, mv_scale, uv_scale, mean = multivariate_model.sample(
        20,
        mv_scale_initial_value=next_mv_state,
        uv_scale_initial_value=next_uv_state,
        mean_initial_value=next_mean_state,
    )

    output_list.append(output)
    mv_scale_list.append(mv_scale)
    uv_scale_list.append(uv_scale)
    mean_list.append(mean)

output = torch.stack(output_list, dim=0)
mv_scale = torch.stack(mv_scale_list, dim=0)
uv_scale = torch.stack(uv_scale_list, dim=0)
mean = torch.stack(mean_list, dim=0)
