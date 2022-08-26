[![test](https://github.com/mawicks/mvarch/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/mawicks/mvarch/actions/workflows/pythonapp.yml)

# Multivariate Volatility Models (ARCH) for stock prices and other time series)

This package uses a simple model for multivariate (and univariate) volatility (ARCH) models

## Motivation

A number of multivariate volatility models exist in the literature (DCC, VECC, BEKK,  etc.).
This package uses a general specification that includes a number of common models.
Using this package you can generate either a DCC model or a BEKK model.

The model predicts the *distribution* of the log returns for the next trading date.
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

### Installation

This package can be installed by running `pip install .` in the top level directory of a `git clone` checkout

     pip install .

### Train a new model on a set of symbols:

Ideally you would train models on a larger set of symbols.  Here we use a small
set for demo purposes:

    python -m mvarch.train_mvarch -s SPY -s QQQ -s BND 


### Evaluate the model on some of the symbols
This script will produce a table and a plot with volatility and mean predictions
for the past and for the trading day.

    TBD



