"""API 키 없이 전체 흐름을 확인하기 위한 가상 데이터. 실제 기업·시장 수치가 아니다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .financials import add_derived
from .market import MarketInputs, last_on_or_before, regression_beta, vol_and_corr

DEMO_NAME = "DEMO항공(가상)"


def demo_history() -> pd.DataFrame:
    years = [2021, 2022, 2023, 2024, 2025]
    h = pd.DataFrame({
        "revenue":             [90000, 130000, 145000, 160000, 165000],
        "operating_income":    [8000, 22000, 12000, 11000, 9000],
        "depreciation":        [20000, 20500, 21000, 22000, 23000],
        "capex":               [9000, 10000, 15000, 18000, 20000],
        "fuel_cost":           [18000, 38000, 42000, 45000, 43000],
        "cash":                [40000, 45000, 42000, 38000, 35000],
        "current_assets":      [70000, 80000, 78000, 76000, 74000],
        "current_liabilities": [90000, 98000, 97000, 99000, 101000],
        "current_debt":        [20000, 21000, 20000, 19000, 18000],
        "current_lease":       [8000, 8000, 8500, 9000, 9500],
        "debt":                [110000, 105000, 100000, 98000, 95000],
        "lease_liabilities":   [45000, 46000, 48000, 50000, 52000],
        "equity_parent":       [70000, 82000, 88000, 92000, 95000],
        "nci":                 [1000, 1000, 1100, 1100, 1200],
    }, index=pd.Index(years, name="year"), dtype=float)
    return add_derived(h)


def _mean_reverting(rng, n: int, level: float, speed: float, vol: float) -> np.ndarray:
    """로그 가격이 level 로 되돌아가는 경로 (가상 시장 데이터용)."""
    x = np.empty(n)
    x[0] = np.log(level)
    for i in range(1, n):
        x[i] = x[i - 1] + speed * (np.log(level) - x[i - 1]) + vol * rng.standard_normal()
    return np.exp(x)


def demo_market(valuation_date: str, seed: int = 7) -> MarketInputs:
    rng = np.random.default_rng(seed)
    val = pd.Timestamp(valuation_date)

    months = pd.date_range(val - pd.DateOffset(years=12), val, freq="MS")
    oil = pd.Series(_mean_reverting(rng, len(months), 75.0, 0.05, 0.30 / np.sqrt(12)), index=months, name="oil")

    days = pd.bdate_range(months[0], val)
    fx = pd.Series(_mean_reverting(rng, len(days), 1330.0, 0.004, 0.08 / np.sqrt(252)), index=days, name="usdkrw")

    z_m = rng.normal(0.0003, 0.012, len(days))
    kospi = pd.Series(2500 * np.exp(np.cumsum(z_m)), index=days, name="kospi")
    stock = pd.Series(np.exp(np.cumsum(1.1 * z_m + rng.normal(0, 0.012, len(days)))), index=days, name="stock")
    stock *= 24000 / stock.iloc[-1]   # 기준일 주가 24,000원이 되도록 수준만 맞춤 (수익률은 그대로)

    lookback = val - pd.DateOffset(months=12)
    oil_vol, fx_vol, corr = vol_and_corr(oil, fx, 120)
    shares = 368_000_000.0
    price = last_on_or_before(stock, val)
    return MarketInputs(
        valuation_date=val, risk_free=0.030, corp_aa=0.034,
        fx_spot=last_on_or_before(fx, val),
        fx_base=float(fx[fx.index > lookback].mean()),
        oil_base=float(oil[oil.index > lookback].mean()),
        oil_vol=oil_vol, fx_vol=fx_vol, oil_fx_corr=corr,
        raw_beta=regression_beta(stock, kospi, 104),
        market_cap=price * shares / 1e8, shares=shares,
        oil_monthly=oil, fx_daily=fx, kospi_daily=kospi, stock_daily=stock,
    )
