"""시장가격 역산: 지금 주가를 설명하려면 어떤 가정이 필요한가.

다른 가정은 그대로 두고 변수 하나만 움직여, DCF 주주가치가 시가총액과 같아지는 값을 찾는다.
해가 구간 안에 없으면(=아무리 움직여도 시장가격에 닿지 않으면) None 을 돌려준다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .model import Assumptions, dcf


def bisect(f, lo: float, hi: float, target: float, iterations: int = 200) -> float | None:
    f_lo, f_hi = f(lo) - target, f(hi) - target
    if f_lo * f_hi > 0:
        return None
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if (f(lo) - target) * (f(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass
class Implied:
    wacc: float | None
    growth: float | None
    nonfuel_ratio: float | None
    oil: float | None
    ev_ebitda_market: float
    ev_ebitda_model: float


def solve_all(a: Assumptions, wacc: float, market_equity: float, base_ebitda: float) -> Implied:
    n = len(a.years)

    def with_nonfuel(x: float) -> float:
        b = copy.copy(a)
        b.nonfuel_ratio = x
        return float(dcf(b, wacc)["equity"][0])

    model = dcf(a, wacc)
    return Implied(
        wacc=bisect(lambda x: float(dcf(a, x)["equity"][0]), a.terminal_growth + 0.001, 0.60, market_equity),
        growth=bisect(lambda g: float(dcf(a, wacc, g=g)["equity"][0]), -0.60, wacc - 0.0005, market_equity),
        nonfuel_ratio=bisect(with_nonfuel, 0.20, 0.95, market_equity),
        oil=bisect(lambda o: float(dcf(a, wacc, oil=np.full(n, o))["equity"][0]), 20.0, 500.0, market_equity),
        ev_ebitda_market=(market_equity + float(model["net_debt"][0]) + a.nci) / base_ebitda,
        ev_ebitda_model=float(model["ev"][0]) / base_ebitda,
    )


def table(imp: Implied, a: Assumptions, wacc: float) -> list[dict]:
    """보고서용 표 (역산값 vs 모델 가정)."""
    def fmt(v, kind):
        if v is None:
            return "해 없음"
        return {"pct": f"{v:.2%}", "num": f"{v:,.0f}", "ratio": f"{v:.1%}"}[kind]
    return [
        {"항목": "할인율(WACC)", "시장가격이 함축하는 값": fmt(imp.wacc, "pct"), "모델 가정": f"{wacc:.2%}"},
        {"항목": "영구성장률", "시장가격이 함축하는 값": fmt(imp.growth, "pct"), "모델 가정": f"{a.terminal_growth:.2%}"},
        {"항목": "비유류 현금비용/매출", "시장가격이 함축하는 값": fmt(imp.nonfuel_ratio, "ratio"), "모델 가정": f"{a.nonfuel_ratio:.1%}"},
        {"항목": "유가(달러/배럴)", "시장가격이 함축하는 값": fmt(imp.oil, "num"), "모델 가정": f"{a.oil_curve[0]:,.0f}"},
        {"항목": "EV/EBITDA(배)", "시장가격이 함축하는 값": f"{imp.ev_ebitda_market:.1f}", "모델 가정": f"{imp.ev_ebitda_model:.1f}"},
    ]
