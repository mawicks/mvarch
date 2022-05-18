# Deep Volatility Models (for stock prices)

This package uses convolutional neural networks (implemented in PyTorch) to train mixture models to model
the volatility of stock prices.

A single model is trained on a number of different stock symbols.  Internally,
an embedding is learned for each symbol.  In other words, a convolutional neural
network learns general features of time series of daily returns that predict the
volatilty along with an embedding that tunes the result for different symbolsl.

## Motivation

The volatility of stock returns changes daily.  The models produced by this
package predict the *distribution* of the log returns for the next trading date.
The actual turn is virtually impossible to predict, but predicting the
distribution of return has several uses:

1. *The distribution can be sampled to generate simulated sequences of returns
   that can be used as synthetic data to test various trading algorithms.*
   Datasets with historic daily returns are very small so testing algorithms using
   historic data is very prone to overfitting.


2. *Knowing the distribution of the daily returns (especially the volatility)
   can be used to determine fair prices for stock options.*  The famous
   Black-Scholes formula predicts fair option prices.  However, it assumes the
   daily returns to be stationary and normally distributed.  However, observed
   daily returns are not stationary (the variance varies with time) and the
   returns are not normally distributed.  They tend to have "long tails"
   compared to a normal distrubution (i.e., kurtosis) and they are not always symmetric
   (i.e., skew).  It's possible to estimate the variances by computing the
   variance of a trailing sample.  However, during periods of increasing
   volatility this would underestimate the volatility since the volatility today
   can be significantly greater than the volatility of the past N days.
   Likewise, during periods of
   decreasing volatility this would overestimate the volatility.  The goal is to
   determine the *instantaneous* volatility to provide estimates of the distribution of
   daily returns during the next trading day (or the next few trading days)

### Train a new model on a set of symbols:

Ideally you would train models on a larger set of symbols.  Here we use a small
set for demo purposes:

    python -m deep_volatility_models.train_univariate -s SPY -s QQQ -s BND 


### Evaluate the model on some of the symbols
This script will produce a table and a plot with volatility and mean predictions
for the past and for the trading day.

    python -m deep)volatility_models.evaluate_model -s SPY -s BND

# Future extensions

Models generated as described above do not model correlations between symsols.
It's possible to generate multivariate models that model the correlations
between symbols.

The inference code described above infers the parameters of a mixture model
representing the distrubtion of daily returns.  No
code has been provided to sample these distributions to generate synthetic data.

