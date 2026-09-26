"""민감도 분석, 유가·환율 몬테카를로, 헤지 비율별 가치 분포."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Assumptions, dcf, timing


def grid_wacc_growth(a: Assumptions, wacc: float, wacc_steps, growth_steps) -> pd.DataFrame:
    """행 = WACC, 열 = 영구성장률, 값 = 주당가치(원)."""
    out = {}
    for dg in growth_steps:
        g = a.terminal_growth + dg
        out[f"{g:.2%}"] = [float(dcf(a, wacc + dw, g)["per_share"][0]) for dw in wacc_steps]
    return pd.DataFrame(out, index=[f"{wacc + dw:.2%}" for dw in wacc_steps])


def grid_oil_fx(a: Assumptions, wacc: float, oil_shocks, fx_shocks) -> pd.DataFrame:
    """행 = 유가 변화, 열 = 환율 변화, 값 = 주당가치(원). 예측기간 전체에 같은 비율로 충격."""
    out = {}
    for fs in fx_shocks:
        col = []
        for os_ in oil_shocks:
            col.append(float(dcf(a, wacc, oil=a.oil_curve * (1 + os_), fx=a.fx_curve * (1 + fs))["per_share"][0]))
        out[f"환율 {fs:+.0%}"] = col
    return pd.DataFrame(out, index=[f"유가 {s:+.0%}" for s in oil_shocks])


def simulate_paths(a: Assumptions, oil_vol: float, fx_vol: float, corr: float, n_paths: int, seed: int):
    """유가·환율의 연평균 가격 경로. 기대값이 기준 커브와 같도록(마팅게일) 로그정규 랜덤워크로 만든다.

    첫해는 평가일 이후 남은 기간만큼만 불확실성이 있다고 보고 분산을 줄인다.
    """
    rng = np.random.default_rng(seed)
    n = len(a.years)
    dt, _, _ = timing(a)
    z1 = rng.standard_normal((n_paths, n))
    z2 = corr * z1 + np.sqrt(1 - corr ** 2) * rng.standard_normal((n_paths, n))
    elapsed = np.cumsum(dt)
    x_oil = np.cumsum(oil_vol * np.sqrt(dt) * z1, axis=1) - 0.5 * oil_vol ** 2 * elapsed
    x_fx = np.cumsum(fx_vol * np.sqrt(dt) * z2, axis=1) - 0.5 * fx_vol ** 2 * elapsed
    return a.oil_curve * np.exp(x_oil), a.fx_curve * np.exp(x_fx)


def hedged_oil(a: Assumptions, oil_paths: np.ndarray, ratio: float, hedge_years: int) -> np.ndarray:
    """앞의 hedge_years 년 동안 물량의 ratio 만큼을 기준 커브(선물가격)로 고정한 실효 유가."""
    eff = oil_paths.copy()
    k = min(hedge_years, eff.shape[1])
    eff[:, :k] = ratio * a.oil_curve[:k] + (1 - ratio) * oil_paths[:, :k]
    return eff


def hedge_analysis(a: Assumptions, wacc: float, oil_paths, fx_paths, ratios, hedge_years: int):
    """헤지 비율별 첫 완전연도 EBITDA 와 주당가치 분포 요약, 그리고 분포 원자료."""
    frac, _, _ = timing(a)
    first_full = int(np.argmax(frac >= 0.999))
    label = f"{a.years[first_full]} EBITDA"
    rows, samples, ebitda = [], {}, {}
    for h in ratios:
        res = dcf(a, wacc, oil=hedged_oil(a, oil_paths, h, hedge_years), fx=fx_paths)
        ebitda1 = res["projection"]["ebitda"][:, first_full]
        ps = res["per_share"]
        samples[h] = ps
        ebitda[h] = ebitda1
        rows.append({
            "헤지비율": f"{h:.0%}",
            f"{label} 평균(억원)": ebitda1.mean(),
            f"{label} 표준편차(억원)": ebitda1.std(),
            f"{label} 적자 확률": (ebitda1 < 0).mean(),
            "주주가치 음수 확률": (ps < 0).mean(),
            "주당가치 평균(원)": ps.mean(),
            "주당가치 P5(원)": np.percentile(ps, 5),
            "주당가치 P50(원)": np.percentile(ps, 50),
            "주당가치 P95(원)": np.percentile(ps, 95),
        })
    return pd.DataFrame(rows), samples, ebitda, label
