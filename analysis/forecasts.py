import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

# Four years of daily data. GARCH and HAR do not forecast before this.
ESTIMATION_DAYS = 1008
REFIT_EVERY = 21
EWMA_LAMBDA = 0.94
ROLLING_WINDOW = 21
ANNUALIZATION = 252
# A 5% annual volatility, written as a daily variance. Forecasts cannot go below this.
VARIANCE_FLOOR = (0.05 ** 2) / ANNUALIZATION


def rolling_forecast(returns):
    # Tomorrow's variance is the sample variance of the last 21 returns.
    # The window ends yesterday, so the day being forecast is not included.
    # Dividing by n-1, rather than by the mean of squared returns, is what was specified.
    variance = returns.rolling(ROLLING_WINDOW).var(ddof=1).shift(1)
    return variance.clip(lower=VARIANCE_FLOOR)


def ewma_forecast(returns):
    # RiskMetrics updates a variance every day: most of yesterday's number, plus a little of yesterday's squared return.
    # The recursion is seeded with the first 21-day sample variance, and that seed is the forecast for the following day.
    # Lambda is 0.94, so the estimate moves slowly. The evaluation sample is long enough that the seed has washed out.
    values = returns.to_numpy(dtype=float)
    n = len(values)
    variance = np.full(n, np.nan)
    variance[ROLLING_WINDOW - 1] = np.var(values[:ROLLING_WINDOW], ddof=1)
    weight = 1.0 - EWMA_LAMBDA
    for t in range(ROLLING_WINDOW, n):
        variance[t] = EWMA_LAMBDA * variance[t - 1] + weight * values[t] ** 2
    # The forecast for day t is the variance dated t-1. Floor only the published forecast.
    forecast = np.full(n, np.nan)
    forecast[1:] = variance[:-1]
    finite = np.isfinite(forecast)
    forecast[finite] = np.maximum(forecast[finite], VARIANCE_FLOOR)
    return pd.Series(forecast, index=returns.index, name="ewma")


def _fit_garch(sample):
    # One Gaussian GARCH(1,1) with a constant mean, on whatever history is passed in.
    # Daily returns are scaled up for the optimizer and the variance is converted back.
    # If the optimizer does not succeed, the caller keeps the parameters from the last fit that did.
    model = arch_model(
        sample,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=True,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)
    except Exception:
        return None
    if not bool(result.optimization_result.success):
        return None
    scale = float(result.scale)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    mu = float(result.params["mu"]) / scale
    omega = float(result.params["omega"]) / (scale ** 2)
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])
    if not np.isfinite([mu, omega, alpha, beta]).all():
        return None
    volatility = np.asarray(result.conditional_volatility, dtype=float)
    residual = np.asarray(result.resid, dtype=float)
    sigma2_last = (volatility[-1] / scale) ** 2
    eps_last = residual[-1] / scale
    if not np.isfinite(sigma2_last) or sigma2_last <= 0.0 or not np.isfinite(eps_last):
        return None
    return mu, omega, alpha, beta, sigma2_last, eps_last


def garch_forecast(returns):
    # Refit every 21 trading days on an expanding window, starting once 1008 days are in hand.
    # Between refits the variance is stepped forward one day at a time with those fixed parameters.
    # The forecast for a day uses the return history through the previous close only.
    values = returns.to_numpy(dtype=float)
    n = len(values)
    forecast = np.full(n, np.nan)
    mu = omega = alpha = beta = None
    last_sigma2 = None
    last_eps = None
    kept = 0
    failed = 0
    print("Fitting GARCH (expanding window, every 21 days)...")
    for t in range(ESTIMATION_DAYS, n):
        if (t - ESTIMATION_DAYS) % REFIT_EVERY == 0:
            fitted = _fit_garch(values[:t])
            if fitted is None:
                failed += 1
                if mu is None:
                    raise RuntimeError("The first GARCH fit failed, so there are no parameters to keep.")
            else:
                mu, omega, alpha, beta, last_sigma2, last_eps = fitted
                kept += 1
        raw = omega + alpha * last_eps ** 2 + beta * last_sigma2
        forecast[t] = raw if raw >= VARIANCE_FLOOR else VARIANCE_FLOOR
        # The filter keeps the model's own variance. The floor is only for the published forecast.
        last_eps = values[t] - mu
        last_sigma2 = raw
    print(f"GARCH refits kept: {kept}. Failed refits: {failed}.")
    return pd.Series(forecast, index=returns.index, name="garch")


def _har_features(squared):
    # Regressors for a forecast of day t, built only from squared returns through t-1.
    # Daily is yesterday's squared return. Weekly and monthly are means of the last 5 and 22.
    # Those means include yesterday, which is the usual overlapping HAR, not a 5-minute realized volatility.
    n = len(squared)
    daily = np.full(n, np.nan)
    weekly = np.full(n, np.nan)
    monthly = np.full(n, np.nan)
    cumulative = np.empty(n + 1)
    cumulative[0] = 0.0
    np.cumsum(squared, out=cumulative[1:])
    for t in range(22, n):
        daily[t] = squared[t - 1]
        weekly[t] = (cumulative[t] - cumulative[t - 5]) / 5.0
        monthly[t] = (cumulative[t] - cumulative[t - 22]) / 22.0
    return daily, weekly, monthly


def har_forecast(returns):
    # OLS of squared returns on a constant, a daily lag, and 5-day and 22-day means.
    # Coefficients are re-estimated every 21 days on the expanding window, same 1008-day start as GARCH.
    # Between refits the coefficients stay put, but the forecast still moves because the lags move.
    squared = returns.to_numpy(dtype=float) ** 2
    daily, weekly, monthly = _har_features(squared)
    n = len(squared)
    forecast = np.full(n, np.nan)
    coef = None
    for t in range(ESTIMATION_DAYS, n):
        if coef is None or (t - ESTIMATION_DAYS) % REFIT_EVERY == 0:
            y = squared[22:t]
            x = np.column_stack(
                [
                    np.ones(t - 22),
                    daily[22:t],
                    weekly[22:t],
                    monthly[22:t],
                ]
            )
            try:
                fitted = sm.OLS(y, x, hasconst=True).fit()
                new_coef = np.asarray(fitted.params, dtype=float)
            except Exception:
                new_coef = None
            if new_coef is not None and np.isfinite(new_coef).all():
                coef = new_coef
            elif coef is None:
                raise RuntimeError("The first HAR fit failed, so there are no coefficients to keep.")
        row = np.array([1.0, daily[t], weekly[t], monthly[t]])
        raw = float(coef @ row)
        forecast[t] = raw if raw >= VARIANCE_FLOOR else VARIANCE_FLOOR
    return pd.Series(forecast, index=returns.index, name="har")


def build_forecasts(returns):
    # Each column is a one-step forecast of next-day simple-return variance.
    # Everything is aligned to the return it is trying to predict, and the floor is already applied.
    # Rolling and EWMA exist before day 1008, but the backtest only scores the shared later dates.
    frame = pd.DataFrame(
        {
            "rolling": rolling_forecast(returns),
            "ewma": ewma_forecast(returns),
            "garch": garch_forecast(returns),
            "har": har_forecast(returns),
        },
        index=returns.index,
    )
    frame.index.name = "date"
    return frame
