"""Forecast losses, strategy statistics, and ranks."""

import numpy as np
import pandas as pd

from analysis.backtest import MODELS

ANNUALIZATION = 252
TARGET_VOL = 0.10
REALIZED_WINDOW = 21
CAP = 1.5

# Calendar slices of the evaluation sample. "other" is whatever is left.
PERIODS = (
    ("2008_2009", "2008-01-01", "2009-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
)


def forecast_scores(forecasts, returns):
    # MSE and QLIKE both compare the variance forecast with the next day's squared return.
    # They are losses, so lower is better. QLIKE is log(forecast) + squared return / forecast.
    # The average is taken only on the evaluation dates passed in.
    squared = returns.to_numpy(dtype=float) ** 2
    rows = []
    for name in forecasts.columns:
        forecast = forecasts[name].to_numpy(dtype=float)
        rows.append(
            {
                "model": name,
                "mse": float(np.mean((forecast - squared) ** 2)),
                "qlike": float(np.mean(np.log(forecast) + squared / forecast)),
            }
        )
    return pd.DataFrame(rows)


def path_metrics(daily, weights, abs_change):
    # Summarize one return path. Annual volatility and Sharpe use the sample standard deviation.
    # The tracking error compares 21-day realized volatility with the 10% target, after dropping
    # the first 20 days, which do not have a full window. Cash earns zero, so Sharpe is not excess of T-bills.
    values = daily.to_numpy(dtype=float)
    if len(values) >= 2:
        vol = float(np.std(values, ddof=1))
        sharpe = float(np.mean(values) / vol * np.sqrt(ANNUALIZATION)) if vol > 0.0 else np.nan
        ann_vol = float(vol * np.sqrt(ANNUALIZATION))
    else:
        ann_vol = np.nan
        sharpe = np.nan
    if len(values) >= REALIZED_WINDOW:
        realized = daily.rolling(REALIZED_WINDOW).std(ddof=1) * np.sqrt(ANNUALIZATION)
        gap = realized.iloc[REALIZED_WINDOW - 1 :].to_numpy(dtype=float) - TARGET_VOL
        tracking = float(np.sqrt(np.mean(gap ** 2)))
    else:
        tracking = np.nan
    wealth = np.cumprod(1.0 + values)
    wealth = np.concatenate([[1.0], wealth])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "ann_vol": ann_vol,
        "tracking_rmse": tracking,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "turnover": float(np.mean(abs_change.to_numpy(dtype=float)) * ANNUALIZATION),
        "sharpe": sharpe,
        "cap_share": float(np.mean(weights.to_numpy(dtype=float) >= CAP - 1e-12)),
    }


def strategy_stats(paths):
    # One row per rule, with net and gross versions of the return-based numbers.
    # Turnover and the share of days at the 1.5 cap depend on the weights, so they are the same either way.
    # Buy-and-hold is included here and left out of the ranks.
    rows = []
    for name, path in paths.items():
        net = path_metrics(path["net"], path["weight"], path["abs_dw"])
        gross = path_metrics(path["gross"], path["weight"], path["abs_dw"])
        rows.append(
            {
                "model": name,
                "ann_vol_net": net["ann_vol"],
                "ann_vol_gross": gross["ann_vol"],
                "tracking_rmse_net": net["tracking_rmse"],
                "tracking_rmse_gross": gross["tracking_rmse"],
                "max_drawdown_net": net["max_drawdown"],
                "max_drawdown_gross": gross["max_drawdown"],
                "turnover": net["turnover"],
                "sharpe_net": net["sharpe"],
                "sharpe_gross": gross["sharpe"],
                "cap_share": net["cap_share"],
            }
        )
    return pd.DataFrame(rows)


def _period_masks(index):
    # Full sample, three stress windows, and the evaluation days that fall in none of them.
    # A short window is still returned. The caller computes it anyway.
    # Dates are the day the return is earned.
    days = pd.DatetimeIndex(index).normalize()
    masks = {"full": np.ones(len(days), dtype=bool)}
    covered = np.zeros(len(days), dtype=bool)
    for name, start, end in PERIODS:
        mask = np.asarray((days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end)))
        masks[name] = mask
        covered |= mask
    masks["other"] = ~covered
    return masks


def rank_table(forecasts, returns, paths):
    # Ranks are among the four forecasts only. 1 is best.
    # QLIKE and the decision metrics are recomputed inside each window, and decision ranks use net returns.
    # Drawdown is ranked so a smaller loss beats a larger one. Ties take the better rank.
    squared_all = returns.to_numpy(dtype=float) ** 2
    masks = _period_masks(returns.index)
    rows = []
    for period, mask in masks.items():
        count = int(mask.sum())
        block = {}
        for name in MODELS:
            forecast = forecasts[name].to_numpy(dtype=float)[mask]
            squared = squared_all[mask]
            qlike = float(np.mean(np.log(forecast) + squared / forecast))
            path = paths[name]
            metrics = path_metrics(
                pd.Series(path["net"].to_numpy(dtype=float)[mask]),
                pd.Series(path["weight"].to_numpy(dtype=float)[mask]),
                pd.Series(path["abs_dw"].to_numpy(dtype=float)[mask]),
            )
            block[name] = {"qlike": qlike, **metrics}
        qlike = pd.Series({name: block[name]["qlike"] for name in MODELS})
        tracking = pd.Series({name: block[name]["tracking_rmse"] for name in MODELS})
        drawdown = pd.Series({name: block[name]["max_drawdown"] for name in MODELS})
        sharpe = pd.Series({name: block[name]["sharpe"] for name in MODELS})
        turnover = pd.Series({name: block[name]["turnover"] for name in MODELS})
        rank_qlike = qlike.rank(ascending=True, method="min")
        rank_tracking = tracking.rank(ascending=True, method="min")
        rank_drawdown = drawdown.rank(ascending=False, method="min")
        rank_sharpe = sharpe.rank(ascending=False, method="min")
        rank_turnover = turnover.rank(ascending=True, method="min")
        for name in MODELS:
            rows.append(
                {
                    "period": period,
                    "model": name,
                    "n_days": count,
                    "qlike": qlike[name],
                    "rank_qlike": rank_qlike[name],
                    "tracking_rmse": tracking[name],
                    "rank_tracking_rmse": rank_tracking[name],
                    "max_drawdown": drawdown[name],
                    "rank_max_drawdown": rank_drawdown[name],
                    "sharpe_net": sharpe[name],
                    "rank_sharpe_net": rank_sharpe[name],
                    "turnover": turnover[name],
                    "rank_turnover": rank_turnover[name],
                }
            )
    table = pd.DataFrame(rows)
    for column in (
        "rank_qlike",
        "rank_tracking_rmse",
        "rank_max_drawdown",
        "rank_sharpe_net",
        "rank_turnover",
    ):
        table[column] = table[column].astype(int)
    return table
