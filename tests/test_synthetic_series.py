import json

import numpy as np
import pytest

from skills.synthetic_series.tools import generate_synthetic_stock_series_garch_arch


def _invoke(**kwargs) -> dict:
    result = generate_synthetic_stock_series_garch_arch.invoke(kwargs)
    return json.loads(result)


def test_simulate_mode_returns_expected_series_shape():
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=10,
        value_type="close",
        n_series=3,
        seed=42,
    )

    assert set(data.keys()) == {"serie_1", "serie_2", "serie_3"}
    for values in data.values():
        assert len(values) == 10
        assert all(isinstance(v, float) for v in values)


def test_simulate_mode_is_deterministic_given_seed():
    first = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        n_series=2,
        seed=7,
    )
    second = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        n_series=2,
        seed=7,
    )

    assert first == second


def test_value_type_selects_the_right_ohlc_field():
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="high",
        n_series=1,
        seed=1,
    )
    close_data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        n_series=1,
        seed=1,
    )

    assert data["serie_1"] != close_data["serie_1"]


def test_fit_and_simulate_mode_estimates_params_from_historical_returns():
    rng = np.random.default_rng(0)
    historical_returns = list(rng.normal(0, 0.015, 60))

    data = _invoke(
        ticker="VALE3",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        n_series=2,
        mode="fit_and_simulate",
        historical_returns=historical_returns,
        seed=1,
    )

    assert set(data.keys()) == {"serie_1", "serie_2"}
    for values in data.values():
        assert len(values) == 5


def test_fit_and_simulate_mode_requires_minimum_historical_returns():
    data = _invoke(
        ticker="VALE3",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        mode="fit_and_simulate",
        historical_returns=[0.01, -0.02, 0.005],
    )

    assert "error" in data


def test_fit_and_simulate_mode_requires_historical_returns():
    data = _invoke(
        ticker="VALE3",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        mode="fit_and_simulate",
    )

    assert "error" in data


@pytest.mark.parametrize("invalid_value_type", ["price", "", "OPEN"])
def test_invalid_value_type_returns_error(invalid_value_type):
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type=invalid_value_type,
    )

    assert "error" in data


def test_invalid_mode_returns_error():
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        mode="not_a_mode",
    )

    assert "error" in data


@pytest.mark.parametrize("n_days", [0, -1])
def test_invalid_n_days_returns_error(n_days):
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=n_days,
        value_type="close",
    )

    assert "error" in data


def test_invalid_n_series_returns_error():
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        n_series=0,
    )

    assert "error" in data


def test_non_stationary_alpha_beta_returns_error():
    data = _invoke(
        ticker="PETR4",
        start_date="2026-01-01",
        n_days=5,
        value_type="close",
        alpha=0.6,
        beta=0.6,
    )

    assert "error" in data
