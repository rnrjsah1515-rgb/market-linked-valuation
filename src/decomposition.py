"""유류비·영업이익 변동 요인 분해.

유류비를 네 요인의 곱으로 본다.

    유류비 = 물량(V) × 항공유가격(J, 달러/배럴) × 환율(F, 원/달러) × 잔차(R)

항공유가격은 미국 걸프연안 현물(FRED DJFUELUSGULF)에 4주 시차를 준 분기평균을 쓴다. 물량은 직접 관측되지 않아
매출을 대용치로 쓰고, 설명되지 않는 부분은 모두 잔차에 남긴다.

    환산 물량 = 유류비 / (J × F)     … 항공유 기준 소비량(배럴)
    잔차 R    = 환산 물량 / 매출      … 연료 효율, 매출-물량 괴리, 매입 시차, 헤지 손익 귀속

원유 대비 항공유 프리미엄(크랙)은 J / 두바이유 로 따로 본다.

로그 변화율로 분해하면 기여가 더해진다.

    Δln(유류비) = Δln(물량) + Δln(항공유가격) + Δln(환율) + Δln(잔차)

전년 동기 대비로 비교해 계절성을 제거한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """원유 환산 소비량과 매출 대비 크랙배수를 계산한다."""
    out = df.copy()
    out["jet_equiv_bbl"] = out["fuel_cost"] * 1e8 / (out["jet_lagged"] * out["fx"])  # 항공유(4주 시차) 환산 물량
    out["oil_equiv_bbl"] = out["fuel_cost"] * 1e8 / (out["oil"] * out["fx"])       # 원유 기준 환산(참고)
    out["fuel_per_revenue"] = out["fuel_cost"] / out["revenue"]
    out["residual_index"] = out["jet_equiv_bbl"] / out["revenue"]                  # 매출 1억원당 환산 물량
    out["crack_ratio"] = out["jet"] / out["oil"]                                   # 항공유 / 원유
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
        jet = ln(cur["jet_lagged"], old["jet_lagged"])
        fx = ln(cur["fx"], old["fx"])
        resid = total - volume - jet - fx
        rows.append({
            "quarter": q,
            "유류비 변동": np.expm1(total),
            "물량(매출) 기여": volume, "항공유가격 기여": jet, "환율 기여": fx, "잔차 기여": resid,
            "크랙 배수": float(cur["crack_ratio"]),
            "합계 확인": volume + jet + fx + resid - total,
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


def fuel_sensitivity(df: pd.DataFrame, price_col: str = "jet_lagged") -> dict:
    """전년 동기 대비 로그 변화율 회귀: Δln(유류비) ~ Δln(가격) + Δln(환율) + Δln(매출).

    price_col="jet" 이면 항공유 가격, "oil" 이면 두바이유를 설명변수로 쓴다.
    항등식상 계수는 1이어야 하므로, 1에서 벗어난 만큼이 잔차 요인과의 동조도다.
    """
    dec = yoy_decomposition(df)
    d = add_factors(df)
    y, x = [], []
    for q in dec.index:
        year, quarter = int(q[:4]), q[-1]
        prev = f"{year - 1}Q{quarter}"
        y.append(np.log(d.loc[q, "fuel_cost"] / d.loc[prev, "fuel_cost"]))
        x.append([1.0,
                  np.log(d.loc[q, price_col] / d.loc[prev, price_col]),
                  np.log(d.loc[q, "fx"] / d.loc[prev, "fx"]),
                  np.log(d.loc[q, "revenue"] / d.loc[prev, "revenue"])])
    y, x = np.array(y), np.array(x)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    r2 = 1 - resid.var() / y.var()
    return {"상수": beta[0], f"{price_col} 가격 계수": beta[1], "환율 계수": beta[2], "매출 계수": beta[3],
            "R2": float(r2), "관측치": len(y)}
