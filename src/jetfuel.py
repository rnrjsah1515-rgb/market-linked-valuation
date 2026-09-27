"""항공유 현물가격 (reference/jet_fuel_spot_usgulf.csv).

FRED 시리즈 DJFUELUSGULF (미국 걸프연안 FOB, 달러/갤런, 일별). 원자료는 EIA.
아시아 기준가격인 싱가포르 MOPS 는 유료 데이터여서 공개된 이 시리즈를 대용치로 쓴다.
갱신 방법은 reference/README.md 참조.
"""
from __future__ import annotations

import pandas as pd

from .config import ROOT

GALLONS_PER_BARREL = 42
PATH = ROOT / "reference" / "jet_fuel_spot_usgulf.csv"


def daily_usd_per_barrel() -> pd.Series:
    df = pd.read_csv(PATH, parse_dates=["observation_date"])
    s = df.set_index("observation_date")["DJFUELUSGULF"].dropna()
    return (s * GALLONS_PER_BARREL).rename("jet")


def quarterly_usd_per_barrel(start: str = "2019-01-01", lag_weeks: int = 0) -> pd.Series:
    """분기 평균 항공유 가격(달러/배럴).

    lag_weeks: 가격을 뒤로 미뤄 매입·소비 시차를 반영한다. 4주가 잔차를 가장 줄였다
    (평균 절대 잔차 14.5%, 시차 없음 17.9%, 두바이유 16.8%).
    """
    s = daily_usd_per_barrel()
    if lag_weeks:
        s = s.copy()
        s.index = s.index + pd.Timedelta(weeks=lag_weeks)
    q = s.resample("QE").mean()
    q = q[q.index >= start]
    q.index = [f"{d.year}Q{d.quarter}" for d in q.index]
    q.index.name = "quarter"
    return q
