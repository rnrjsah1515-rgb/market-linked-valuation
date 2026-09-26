"""유류비·영업이익 변동 요인 분해.

유류비는 네 가지가 곱해진 값으로 본다.

    유류비 = 물량(V) × 원유가격(P, 달러/배럴) × 환율(F, 원/달러) × 크랙배수(C)

V 와 C 를 따로 관측할 수 없으므로, 물량 대용치로 **매출(실질)** 을 쓰고 나머지를 크랙배수에 남긴다.

    C·V = 유류비 / (P × F)          … 원유 환산 소비량
    V   = 매출 / 매출단가 대용치     … 여기서는 매출 자체를 물량 대용치로 사용
    C   = (C·V) / V                 … 원유 대비 항공유·효율 요인

로그 변화율로 분해하면 각 요인의 기여가 더해진다.

    Δln(유류비) = Δln(물량) + Δln(원유가격) + Δln(환율) + Δln(크랙배수)

전년 동기 대비로 비교해 계절성을 제거한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """원유 환산 소비량과 매출 대비 크랙배수를 계산한다."""
    out = df.copy()
    out["oil_equiv_bbl"] = out["fuel_cost"] * 1e8 / (out["oil"] * out["fx"])       # 배럴
    out["fuel_per_revenue"] = out["fuel_cost"] / out["revenue"]
    out["crack_index"] = out["oil_equiv_bbl"] / out["revenue"]                     # 매출 1억원당 원유환산 배럴
    return out


def yoy_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """전년 동기 대비 유류비 변동을 물량·원유가격·환율·크랙 요인으로 나눈다 (로그 기여도, %p)."""
    d = add_factors(df)
    idx = list(d.index)
    rows = []
    for q in idx:
        year, quarter = int(q[:4]), q[-1]
        prev = f"{year - 1}Q{quarter}"
        if prev not in d.index:
            continue
        cur, old = d.loc[q], d.loc[prev]
        ln = lambda a, b: float(np.log(a / b))
        total = ln(cur["fuel_cost"], old["fuel_cost"])
        volume = ln(cur["revenue"], old["revenue"])
        oil = ln(cur["oil"], old["oil"])
        fx = ln(cur["fx"], old["fx"])
        crack = total - volume - oil - fx
        rows.append({
            "quarter": q,
            "유류비 변동": np.expm1(total),
            "물량(매출) 기여": volume, "원유가격 기여": oil, "환율 기여": fx, "크랙·효율 기여": crack,
            "합계 확인": volume + oil + fx + crack - total,
        })
    return pd.DataFrame(rows).set_index("quarter")


def operating_bridge(df: pd.DataFrame) -> pd.DataFrame:
    """전년 동기 대비 영업이익 변동을 매출·유류비·기타비용으로 나눈다 (억원)."""
    rows = []
    for q in df.index:
        year, quarter = int(q[:4]), q[-1]
        prev = f"{year - 1}Q{quarter}"
        if prev not in df.index:
            continue
        cur, old = df.loc[q], df.loc[prev]
        other_cur = cur["revenue"] - cur["fuel_cost"] - cur["operating_income"]
        other_old = old["revenue"] - old["fuel_cost"] - old["operating_income"]
        rows.append({
            "quarter": q,
            "영업이익 변동": cur["operating_income"] - old["operating_income"],
            "매출 증감": cur["revenue"] - old["revenue"],
            "유류비 증감(−)": -(cur["fuel_cost"] - old["fuel_cost"]),
            "기타비용 증감(−)": -(other_cur - other_old),
        })
    return pd.DataFrame(rows).set_index("quarter")


def fuel_sensitivity(df: pd.DataFrame) -> dict:
    """전년 동기 대비 로그 변화율 회귀: Δln(유류비) ~ Δln(원유) + Δln(환율) + Δln(매출)."""
    dec = yoy_decomposition(df)
    d = add_factors(df)
    y, x = [], []
    for q in dec.index:
        year, quarter = int(q[:4]), q[-1]
        prev = f"{year - 1}Q{quarter}"
        y.append(np.log(d.loc[q, "fuel_cost"] / d.loc[prev, "fuel_cost"]))
        x.append([1.0,
                  np.log(d.loc[q, "oil"] / d.loc[prev, "oil"]),
                  np.log(d.loc[q, "fx"] / d.loc[prev, "fx"]),
                  np.log(d.loc[q, "revenue"] / d.loc[prev, "revenue"])])
    y, x = np.array(y), np.array(x)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    r2 = 1 - resid.var() / y.var()
    return {"상수": beta[0], "원유가격 탄력성": beta[1], "환율 탄력성": beta[2], "매출 탄력성": beta[3],
            "R2": float(r2), "관측치": len(y)}
