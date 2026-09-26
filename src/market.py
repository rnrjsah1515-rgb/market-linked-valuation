"""시장 데이터로 할인율·유가·환율 가정을 만든다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketInputs:
    valuation_date: pd.Timestamp
    risk_free: float               # 국고채 10년 (소수)
    corp_aa: float                 # 회사채 3년 AA- (소수)
    fx_spot: float                 # 원/달러 기준일
    fx_base: float                 # 원/달러 최근 N개월 평균 (기준 환율 가정)
    oil_base: float                # 유가 최근 N개월 평균 (달러/배럴)
    oil_vol: float                 # 연환산 변동성
    fx_vol: float
    oil_fx_corr: float
    raw_beta: float | None
    market_cap: float | None       # 억원
    shares: float | None
    oil_monthly: pd.Series
    fx_daily: pd.Series
    kospi_daily: pd.Series | None = None
    stock_daily: pd.Series | None = None
    beta_index_name: str = "KOSPI"   # 베타 추정에 쓴 시장지수
    beta_index: pd.Series | None = None
    semi_weight: float = 0.0         # KOSPI 내 반도체 3종목 비중 (참고)
    beta_r2: float = 0.0             # 베타 회귀분석 설명력

    def fx_annual_avg(self, year: int) -> float:
        return float(self.fx_daily[self.fx_daily.index.year == year].mean())

    def oil_annual_avg(self, year: int) -> float:
        return float(self.oil_monthly[self.oil_monthly.index.year == year].mean())


def last_on_or_before(s: pd.Series, date: pd.Timestamp) -> float:
    s = s[s.index <= date].dropna()
    if s.empty:
        raise ValueError(f"{s.name}: {date.date()} 이전 데이터가 없습니다.")
    return float(s.iloc[-1])


def vol_and_corr(oil_monthly: pd.Series, fx_daily: pd.Series, months: int) -> tuple[float, float, float]:
    """월간 로그수익률로 연환산 변동성과 상관계수를 구한다."""
    fx_m = fx_daily.resample("MS").mean()
    df = pd.concat([oil_monthly.rename("oil"), fx_m.rename("fx")], axis=1).dropna().tail(months + 1)
    r = np.log(df).diff().dropna()
    return float(r["oil"].std() * np.sqrt(12)), float(r["fx"].std() * np.sqrt(12)), float(r["oil"].corr(r["fx"]))


def regression_beta(stock: pd.Series, index: pd.Series, weeks: int) -> tuple[float, float, int]:
    """주간(금요일 종가) 수익률 회귀 → (베타, 결정계수 R², 관측치 수)."""
    df = pd.concat([stock.rename("s"), index.rename("m")], axis=1).dropna()
    weekly = df.resample("W-FRI").last().dropna()
    r = weekly.pct_change().dropna().tail(weeks)
    if len(r) < 26:
        raise ValueError("베타 추정에 필요한 주간 수익률이 26개 미만입니다.")
    cov = np.cov(r["s"], r["m"], ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])
    r2 = float(np.corrcoef(r["s"], r["m"])[0, 1] ** 2)
    return beta, r2, len(r)


def blume_adjust(beta: float) -> float:
    return 0.67 * beta + 0.33


# ── 기초자산 가격 커브 ─────────────────────────────────────────
class FlatCurve:
    """선물 커브가 없을 때: 최근 N개월 평균 가격이 향후에도 유지된다고 가정한다."""

    source = "최근 평균가격 고정(선물 커브 미반영)"

    def __init__(self, oil: float, fx: float):
        self.oil, self.fx = oil, fx

    def oil_curve(self, years: list[int]) -> np.ndarray:
        return np.full(len(years), self.oil)

    def fx_curve(self, years: list[int]) -> np.ndarray:
        return np.full(len(years), self.fx)


class NHFuturesCurve:
    """NH선물 REST API 의 선물 가격으로 연도별 선도가격을 만든다. (프로그램 참여 후 구현 예정)

    구현 계획:
      - 해외선물: 원유 월물별 결제가격 → 회계연도별 평균 선도가격
      - 국내선물: 미국달러선물 월물 → 연도별 선도환율 (만기 이후 구간은 금리평형으로 외삽)
      - 선물 만기가 예측기간보다 짧으면 마지막 월물 가격을 유지하고 그 사실을 보고서에 표시
    """

    source = "NH선물 선물 커브"

    def __init__(self, *_, **__):
        raise NotImplementedError("NH선물 API 연동은 프로그램 참여 후 구현합니다. 지금은 FlatCurve 를 사용하세요.")
