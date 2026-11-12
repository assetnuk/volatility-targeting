"""Load SPY, forecast variance, and run one volatility-targeting rule."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.backtest import run_backtest
from analysis.forecasts import ESTIMATION_DAYS, build_forecasts
from analysis.metrics import forecast_scores, rank_table, strategy_stats

ROOT = Path(__file__).resolve().parent
PRICE_PATH = ROOT / "data" / "spy.csv"
OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

COLORS = {
    "rolling": "C0",
    "ewma": "C1",
    "garch": "C2",
    "har": "C3",
    "buy_and_hold": "black",
}


def load_prices():
    # Use the cached adjusted close when the file is already there and has rows.
    # Otherwise download the full SPY history from Yahoo Finance and write date, adj_close.
    # A second attempt is the only retry. Nothing is filled in if both attempts fail.
    if PRICE_PATH.exists() and PRICE_PATH.stat().st_size > 0:
        cached = pd.read_csv(PRICE_PATH, parse_dates=["date"])
        if len(cached) > 0 and {"date", "adj_close"}.issubset(cached.columns):
            cached = cached.dropna(subset=["adj_close"]).drop_duplicates("date", keep="last")
            return cached.sort_values("date")
    raw = _download_spy()
    close = _adjusted_close(raw)
    frame = pd.DataFrame({"adj_close": close.astype(float)})
    frame.index = pd.to_datetime(close.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame.index.name = "date"
    frame = frame.reset_index().dropna(subset=["adj_close"])
    frame = frame[frame["adj_close"] > 0].drop_duplicates("date", keep="last").sort_values("date")
    PRICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = frame.copy()
    saved["date"] = saved["date"].dt.strftime("%Y-%m-%d")
    saved.to_csv(PRICE_PATH, index=False)
    return frame


def _download_spy():
    # Pull the whole SPY history. Adjusted close is a column, not a back-adjusted invention.
    # The second try is only there for a transient Yahoo failure.
    # The error from the second failure is reported rather than replaced with made-up prices.
    import yfinance as yf

    last_error = None
    for _attempt in range(2):
        try:
            raw = yf.download(
                "SPY",
                period="max",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is None or len(raw) == 0:
                raise RuntimeError("yfinance returned no SPY prices")
            return raw
        except Exception as err:
            last_error = err
    raise RuntimeError(f"yfinance failed after two attempts: {last_error}") from last_error


def _adjusted_close(raw):
    # yfinance sometimes returns a flat table and sometimes a two-level column index.
    # Either way, the series we want is the adjusted close, not the raw close.
    # The raw close is only a fallback if the adjusted column is missing.
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        key = "Adj Close" if "Adj Close" in fields else "Close"
        series = raw[key]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    elif "Adj Close" in raw.columns:
        series = raw["Adj Close"]
    else:
        series = raw["Close"]
    series = series.dropna().copy()
    series.name = "adj_close"
    return series


def simple_returns(prices):
    # Simple return from the adjusted close: today's price divided by yesterday's, minus one.
    # The date is the day the return is earned. The first price has no return, so it is dropped.
    # Later, the forecast for that date is built only from earlier returns.
    level = prices.set_index("date")["adj_close"].astype(float).sort_index()
    returns = level.pct_change().dropna()
    returns.name = "spy"
    returns.index.name = "date"
    return returns


def save_figure(fig, path):
    # Write the figure under output/figures. The README points at these relative paths.
    # tight_layout keeps the axis labels from being cut off.
    # The script uses a non-interactive backend, so this does not open a window.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forecasts(forecasts, returns, path):
    # Daily variance forecasts are too jagged to read, so each line is a 63-day mean.
    # Squared returns are the series the MSE and QLIKE scores are computed against.
    # Only the evaluation sample is drawn.
    panel = forecasts.copy()
    panel["squared return"] = returns ** 2
    smooth = panel.rolling(63, min_periods=63).mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for name in forecasts.columns:
        ax.plot(smooth.index, smooth[name], label=name, color=COLORS[name], lw=1.2)
    ax.plot(
        smooth.index,
        smooth["squared return"],
        label="squared return",
        color="black",
        lw=1.15,
        alpha=0.75,
    )
    ax.set_title("Variance forecasts and squared returns (63-day mean)")
    ax.set_ylabel("63-day mean of daily variance")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    save_figure(fig, path)


def plot_wealth(paths, path):
    # Cumulative net wealth of the four rules and of buy-and-hold, starting at 1.
    # Costs are already in the net return. Buy-and-hold has none.
    # The dashed line is the reference, not a competing forecast.
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for name in ["rolling", "ewma", "garch", "har", "buy_and_hold"]:
        net = paths[name]["net"]
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + net.to_numpy(dtype=float))])
        start = net.index[0] - pd.tseries.offsets.BDay(1)
        plot_index = net.index.insert(0, start)
        label = "buy and hold" if name == "buy_and_hold" else name
        ax.plot(
            plot_index,
            wealth,
            label=label,
            color=COLORS[name],
            lw=1.2,
            ls="--" if name == "buy_and_hold" else "-",
        )
    ax.set_title("Cumulative wealth, net of costs")
    ax.set_ylabel("Wealth (start = 1)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    save_figure(fig, path)


def weight_plot_models(scores, stats):
    # The weight chart shows the best QLIKE forecast and the best net-Sharpe forecast.
    # If those are the same model, EWMA is the second line.
    # If that winner is already EWMA, the second line is the next-best net Sharpe instead.
    qlike = scores.set_index("model")["qlike"]
    sharpe = stats.set_index("model")["sharpe_net"].drop(labels=["buy_and_hold"])
    qlike_best = qlike.idxmin()
    sharpe_best = sharpe.idxmax()
    if qlike_best != sharpe_best:
        names = [qlike_best, sharpe_best]
        title = f"Weights: {qlike_best} (lowest QLIKE) and {sharpe_best} (highest net Sharpe)"
        return names, title
    if qlike_best == "ewma":
        second = sharpe.drop(labels=["ewma"]).idxmax()
        names = ["ewma", second]
        title = (
            "Weights: ewma (lowest QLIKE and highest net Sharpe) "
            f"and {second} (next-best net Sharpe)"
        )
        return names, title
    names = [qlike_best, "ewma"]
    title = f"Weights: {qlike_best} (lowest QLIKE and highest net Sharpe) and ewma"
    return names, title


def plot_weights(paths, names, title, path):
    # Weights over the evaluation sample for the two models chosen above.
    # The dotted line is the 1.5 cap. A weight at that line is fully at the cap.
    # The rule never shorts, so the lines stay between zero and the cap.
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for name in names:
        ax.plot(paths[name].index, paths[name]["weight"], label=name, color=COLORS[name], lw=1.0)
    ax.axhline(1.5, color="gray", lw=0.8, ls=":", label="cap 1.5")
    ax.set_ylim(0.0, 1.7)
    ax.set_title(title)
    ax.set_ylabel("Weight in SPY")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    save_figure(fig, path)


def main():
    # Cache or download prices, build the four forecasts, and backtest the fixed rule.
    # The first 1008 return days are estimation only. Scores, ranks, and plots use the later days.
    # Tables go to output/ and the three figures go to output/figures/.
    prices = load_prices()
    returns = simple_returns(prices)
    if len(returns) <= ESTIMATION_DAYS:
        raise RuntimeError("Not enough SPY history to leave an evaluation sample after 1008 days.")
    forecasts = build_forecasts(returns)
    evaluation = returns.iloc[ESTIMATION_DAYS:]
    forecasts = forecasts.loc[evaluation.index]
    if not forecasts.notna().all().all():
        raise RuntimeError("A forecast is missing on the evaluation sample.")
    paths = run_backtest(forecasts, evaluation)
    scores = forecast_scores(forecasts, evaluation)
    stats = strategy_stats(paths)
    ranks = rank_table(forecasts, evaluation, paths)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "forecast_scores.csv", index=False, float_format="%.10g")
    stats.to_csv(OUTPUT_DIR / "strategy_stats.csv", index=False, float_format="%.10g")
    ranks.to_csv(OUTPUT_DIR / "ranks.csv", index=False, float_format="%.10g")

    plot_forecasts(forecasts, evaluation, FIGURE_DIR / "forecast_vs_realized.png")
    plot_wealth(paths, FIGURE_DIR / "wealth.png")
    names, title = weight_plot_models(scores, stats)
    plot_weights(paths, names, title, FIGURE_DIR / "weights.png")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda value: f"{value:.6g}")
    print("\nForecast scores")
    print(scores.to_string(index=False))
    print("\nStrategy stats")
    print(stats.to_string(index=False))
    print("\nRanks")
    print(ranks.to_string(index=False))
    start = evaluation.index[0].date()
    end = evaluation.index[-1].date()
    print(f"\nEvaluation: {start} to {end} ({len(evaluation)} days)")
    print(f"Weight plot: {', '.join(names)}")


if __name__ == "__main__":
    main()
