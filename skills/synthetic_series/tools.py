import json
from datetime import datetime, timedelta
from typing import Annotated

import numpy as np
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from langchain_core.tools import tool


def _business_days_from(start: datetime, n_days: int) -> list[datetime]:
    dates = []
    day = start
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def _returns_to_ohlc(
    returns_pct: np.ndarray,
    dates: list[datetime],
    initial_price: float,
    rng: np.random.Generator,
) -> list[dict]:
    points = []
    price = initial_price
    for i, date in enumerate(dates):
        ret = returns_pct[i] / 100.0
        day_open = price
        day_close = day_open * (1 + ret)
        intraday_vol = abs(ret) / 2 + 1e-4
        high_extra = abs(rng.normal(0, intraday_vol))
        low_extra = abs(rng.normal(0, intraday_vol))
        day_high = max(day_open, day_close) * (1 + high_extra)
        day_low = min(day_open, day_close) * (1 - low_extra)
        points.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(day_open, 2),
            "high": round(day_high, 2),
            "low": round(day_low, 2),
            "close": round(day_close, 2),
        })
        price = day_close
    return points


def _fit_garch_params(historical_returns: list[float]) -> tuple[float, float, float, float]:
    hist_pct = np.array(historical_returns) * 100.0
    model = arch_model(hist_pct, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")
    return (
        float(result.params["mu"]),
        float(result.params["omega"]),
        float(result.params["alpha[1]"]),
        float(result.params["beta[1]"]),
    )


@tool
def generate_synthetic_stock_series_garch_arch(
    ticker: Annotated[str, "Stock ticker, for example PETR4 or AAPL. Used in messages only; no market data is fetched."],
    start_date: Annotated[str, "First calendar date of the series, in YYYY-MM-DD format. Weekends are skipped."],
    n_days: Annotated[int, "Number of business days to simulate. Must be at least 1."],
    value_type: Annotated[str, "OHLC field to return for each series: 'open', 'high', 'low' or 'close'."] = "close",
    n_series: Annotated[int, "How many independent price paths to generate. Must be at least 1."] = 1,
    mode: Annotated[str, "'simulate' uses the GARCH parameters below; 'fit_and_simulate' estimates them from historical_returns first."] = "simulate",
    initial_price: Annotated[float, "Price of the asset on the day before the first simulated day."] = 100.0,
    mu: Annotated[float, "Constant mean of the daily return, in percent (0.1 means 0.1% per day). Used in 'simulate' mode."] = 0.0,
    omega: Annotated[float, "GARCH constant term for the conditional variance, in percent squared. Used in 'simulate' mode."] = 0.02,
    alpha: Annotated[float, "GARCH ARCH coefficient: how strongly yesterday's shock moves today's volatility. Must satisfy alpha + beta < 1."] = 0.10,
    beta: Annotated[float, "GARCH GARCH coefficient: how persistent yesterday's volatility is. Must satisfy alpha + beta < 1."] = 0.85,
    historical_returns: Annotated[list[float] | None, "Past daily fractional returns (0.01 means 1%), at least 30 values. Required in 'fit_and_simulate' mode."] = None,
    seed: Annotated[int | None, "Seed of the random generator, for reproducible series. Omit for a different result on each call."] = None,
) -> str:
    """Generate synthetic daily price series with a GARCH(1,1) model.

    Each series is a path of business days (weekends skipped) with open, high,
    low and close prices derived from simulated daily returns. Use it to
    illustrate plausible future price paths or to stress-test strategies
    against synthetic scenarios.

    The result is a statistical simulation, not a forecast. It does not
    constitute investment advice.

    Parameters
    ----------
    ticker : str
        Stock ticker, used in error messages only.
    start_date : str
        First calendar date, in YYYY-MM-DD format.
    n_days : int
        Number of business days to simulate.
    value_type : str
        Which OHLC field to return: 'open', 'high', 'low' or 'close'.
    n_series : int
        Number of independent price paths.
    mode : str
        'simulate' or 'fit_and_simulate'.
    initial_price : float
        Starting price, before the first simulated day.
    mu, omega, alpha, beta : float
        GARCH(1,1) parameters used in 'simulate' mode. alpha + beta must be
        below 1 for the process to be stationary.
    historical_returns : list[float] or None
        Daily fractional returns used to estimate the parameters in
        'fit_and_simulate' mode (at least 30 values).
    seed : int or None
        Seed for reproducible output.

    Returns
    -------
    str
        JSON object with one array per series, named ``serie_1`` to
        ``serie_n``. On invalid input, a JSON object with an ``error`` key.

    Example
    -------
    >>> generate_synthetic_stock_series_garch_arch.invoke(  # doctest: +SKIP
    ...     {"ticker": "PETR4", "start_date": "2026-01-01", "n_days": 5, "seed": 7}
    ... )
    '{"serie_1": [...]}'
    """
    if value_type not in ("open", "high", "low", "close"):
        return json.dumps({"error": "value_type must be 'open', 'high', 'low' or 'close'."})
    if mode not in ("simulate", "fit_and_simulate"):
        return json.dumps({"error": "mode must be 'simulate' or 'fit_and_simulate'."})
    if n_days < 1:
        return json.dumps({"error": "n_days must be >= 1."})
    if n_series < 1:
        return json.dumps({"error": "n_series must be >= 1."})

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = _business_days_from(start, n_days)

        if mode == "fit_and_simulate":
            if not historical_returns or len(historical_returns) < 30:
                return json.dumps({
                    "error": "mode='fit_and_simulate' requires 'historical_returns' with at least 30 observations."
                })
            mu_, omega_, alpha_, beta_ = _fit_garch_params(historical_returns)
        else:
            if alpha + beta >= 1:
                return json.dumps({"error": "alpha + beta must be < 1 (GARCH stationarity)."})
            mu_, omega_, alpha_, beta_ = mu, omega, alpha, beta

        rng_master = np.random.default_rng(seed)
        series_output = {}
        for series_idx in range(n_series):
            series_seed = int(rng_master.integers(0, 2**31 - 1))
            rng_local = np.random.default_rng(series_seed)

            sim_model = ConstantMean()
            sim_model.volatility = GARCH(p=1, q=1)
            sim_model.distribution = Normal(seed=series_seed)
            sim_data = sim_model.simulate([mu_, omega_, alpha_, beta_], n_days)

            returns_pct = sim_data["data"].values
            ohlc_points = _returns_to_ohlc(returns_pct, dates, initial_price, rng_local)

            series_output[f"serie_{series_idx + 1}"] = [point[value_type] for point in ohlc_points]

        return json.dumps(series_output, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Could not generate synthetic series for {ticker}. Reason: {str(e)}"})
