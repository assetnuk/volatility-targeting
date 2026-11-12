"""Turn a variance forecast into a position and a portfolio return."""

import numpy as np
import pandas as pd

ANNUALIZATION = 252
TARGET_VOL = 0.10
WEIGHT_CAP = 1.5
# One basis point of the absolute change in weight.
COST = 0.0001
MODELS = ["rolling", "ewma", "garch", "har"]


def strategy_path(variance, returns):
    # The weight is known at yesterday's close and is applied to today's simple return.
    # It is 10% divided by the forecast annual volatility, and it cannot exceed 1.5 or go short.
    # The first trade starts from a flat position. The cost is one basis point times the weight change.
    annual = np.sqrt(variance.to_numpy(dtype=float) * ANNUALIZATION)
    weight = np.minimum(WEIGHT_CAP, TARGET_VOL / annual)
    simple = returns.to_numpy(dtype=float)
    previous = np.concatenate([[0.0], weight[:-1]])
    abs_change = np.abs(weight - previous)
    gross = weight * simple
    net = gross - COST * abs_change
    return pd.DataFrame(
        {"weight": weight, "gross": gross, "net": net, "abs_dw": abs_change},
        index=returns.index,
    )


def buy_and_hold(returns):
    # Buy-and-hold is a reference, not a fifth forecast. The weight stays at 1.
    # There is no trading cost. It is not included when the four forecasts are ranked.
    n = len(returns)
    simple = returns.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "weight": np.ones(n),
            "gross": simple,
            "net": simple,
            "abs_dw": np.zeros(n),
        },
        index=returns.index,
    )


def run_backtest(forecasts, returns):
    # Run the same rule on each forecast, plus the buy-and-hold reference.
    # Gross returns set the cost to zero. Net returns subtract it.
    # Cash earns zero, so the uninvested part of the book adds nothing.
    paths = {name: strategy_path(forecasts[name], returns) for name in MODELS}
    paths["buy_and_hold"] = buy_and_hold(returns)
    return paths
