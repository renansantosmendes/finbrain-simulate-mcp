import json
from datetime import datetime, timedelta

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
    ticker: str,
    start_date: str,
    n_days: int,
    value_type: str = "close",
    n_series: int = 1,
    mode: str = "simulate",
    initial_price: float = 100.0,
    mu: float = 0.0,
    omega: float = 0.02,
    alpha: float = 0.10,
    beta: float = 0.85,
    historical_returns: list[float] | None = None,
    seed: int | None = None,
) -> str:
    """
    Generate N synthetic price time series for a stock using a GARCH(1,1) model
    (`arch` package). Useful to simulate plausible future price paths or to
    stress-test strategies against synthetic scenarios.

    value_type selects which OHLC field is returned ('open', 'high', 'low', 'close').
    mode='simulate' uses the manually informed mu/omega/alpha/beta parameters.
    mode='fit_and_simulate' estimates GARCH(1,1) parameters via MLE from
    'historical_returns' (daily fractional returns, at least 30 observations)
    and simulates using the fitted parameters.

    Returns a JSON string shaped as {"serie_1": [values...], ..., "serie_n": [values...]}.
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
