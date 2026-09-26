"""과거 시점 검증: 그 시점에 공시된 자료만으로 모델을 돌렸다면 주가와 얼마나 달랐는가.

평가일은 각 사업보고서 공시 직후(3월 말)로 잡는다. 그 시점에 알 수 있는 것은 직전 연도 실적,
그때까지의 유가·환율·금리, 그날의 주가뿐이다.

실행: python backtest.py
"""
from __future__ import annotations

import copy

import numpy as np

import pandas as pd

from src.config import ROOT, load_config, load_env
from src.market import FlatCurve
from src import report
from src.implied import bisect
from src.model import build_assumptions, compute_wacc, dcf
from src.config import api_key
from src.pipeline import load_financials, load_market
from src.sources import datagokr, ecos

# 기준연도 → (평가일, 그 해 말 순외화부채(달러))
# 근거: 각 연도 사업보고서 'II. 사업의 내용 > 시장위험' 의 순외화부채 공시
CASES = {
    2020: ("2021-03-31", 7.0e9),
    2021: ("2022-03-31", 4.5e9),
    2022: ("2023-03-31", 3.0e9),
    2023: ("2024-03-31", 2.7e9),
    2024: ("2025-03-31", 3.5e9),
}
ADOPTED_WACC = 0.084   # 회사 공시값 (2024 사업보고서 주석 46). 과거 시점에도 같은 값을 적용한다.


def _forward_oil(c: dict, val: pd.Timestamp) -> float:
    """평가일 이후 1년간 실제 유가(월평균) — 모델이 볼 수 없었던 정보."""
    end = val + pd.DateOffset(years=1)
    s = ecos.get_oil_monthly(api_key("ECOS_API_KEY"), c["market_data"]["oil_benchmark"],
                             val.strftime("%Y%m"), end.strftime("%Y%m"))
    return float(s[(s.index > val) & (s.index <= end)].mean())


def run_case(cfg: dict, base_year: int, valuation_date: str, usd_net_debt: float) -> dict:
    c = copy.deepcopy(cfg)
    # 사이클 평균을 내려면 과거가 필요하다. 유류비 주석을 확보한 2019년부터 기준연도까지 쓴다.
    c["company"]["history_years"] = list(range(max(2019, base_year - 4), base_year + 1))
    c["company"]["valuation_date"] = valuation_date
    c["company"].pop("latest_balance_sheet", None)
    c["forecast"]["ratio_years"] = 1
    c["exposure"]["usd_net_debt"] = usd_net_debt
    # pro forma 합산은 재무상태표에 아시아나가 들어온 2024년에만 적용한다
    c["proforma"] = {k: v for k, v in c.get("proforma", {}).items() if k == str(base_year) and base_year == 2024}

    hist, _, _ = load_financials(c)
    m = load_market(c)
    a = build_assumptions(c, hist, m, FlatCurve(m.oil_base, m.fx_base), m.shares)
    a.fx_at_balance_sheet = float(m.fx_daily[m.fx_daily.index <= pd.Timestamp(base_year, 12, 31)].iloc[-1])
    w = compute_wacc(c, m, float(hist["debt"].iloc[-1] + hist["lease_liabilities"].iloc[-1]))

    price = float(m.stock_daily[m.stock_daily.index <= m.valuation_date].iloc[-1])
    # 평가일 이후 1년 주가(사후 확인용) — 모델이 볼 수 없었던 정보
    fwd = datagokr.stock_prices(api_key("DATA_GO_KR_API_KEY"), c["company"]["stock_code"],
                                m.valuation_date.strftime("%Y%m%d"),
                                (m.valuation_date + pd.DateOffset(years=1)).strftime("%Y%m%d"))
    price_1y = float(fwd["close"].iloc[-1])
    # 마진 정상화 변형: (1) 사이클 평균 마진, (2) 기준연도 마진 → 사이클 평균으로 3년에 걸쳐 수렴
    fuel_hist = hist.dropna(subset=["fuel_cost"])
    cycle = float((fuel_hist["nonfuel_cash_opex"] / fuel_hist["revenue"]).mean())
    n = len(a.years)
    a_cycle = copy.copy(a)
    a_cycle.nonfuel_ratio = cycle
    a_fade = copy.copy(a)
    fade_years = min(3, n)
    a_fade.nonfuel_ratio = np.concatenate([
        np.linspace(float(a.nonfuel_ratio), cycle, fade_years + 1)[1:],
        np.full(n - fade_years, cycle),
    ])

    # (3) 매출 수준까지 정상화: 최근 3년 평균 매출을 기준으로 두고 유류 소모량도 같은 비율로 조정
    norm_rev = float(hist["revenue"].tail(3).mean())
    a_norm = copy.copy(a)
    a_norm.nonfuel_ratio = cycle
    a_norm.revenue_last = norm_rev
    a_norm.fuel_volume_last = a.fuel_volume_last * norm_rev / a.revenue_last
    a_norm.nwc_last = a.nwc_last

    r = dcf(a, ADOPTED_WACC)
    r_cycle, r_fade = dcf(a_cycle, ADOPTED_WACC), dcf(a_fade, ADOPTED_WACC)
    r_norm = dcf(a_norm, ADOPTED_WACC)

    # 시기별 시장위험 역산: 그날 주가를 설명하는 할인율(=시장이 요구한 수익률)
    lo = a.terminal_growth + 0.001
    implied_wacc = {k: bisect(lambda x, aa=aa: float(dcf(aa, x)["equity"][0]), lo, 0.60, m.market_cap)
                    for k, aa in (("기준연도", a), ("사이클평균", a_cycle), ("마진수렴", a_fade), ("매출+마진", a_norm))}
    # 할인율 → 자기자본비용 → 시장위험프리미엄 역산 (자본구조·타인자본비용·베타는 그 시점 관측값)
    iw = implied_wacc["마진수렴"]
    implied_ke = (iw - w.debt_weight * w.after_tax_kd) / w.equity_weight if iw else None
    implied_erp = (implied_ke - w.risk_free) / w.relevered_beta if implied_ke else None
    r_capm = dcf(a, w.capm_wacc) if w.capm_wacc > a.terminal_growth else None
    ebitda = float(hist["ebitda"].iloc[-1])
    return {
        "평가일": valuation_date, "기준연도": base_year,
        "EBITDA(억원)": ebitda,
        "EBITDA 마진": float(hist["ebitda"].iloc[-1] / hist["revenue"].iloc[-1]),
        "비유류비/매출": a.nonfuel_ratio,
        "기준 유가": float(a.oil_curve[0]), "기준 환율": float(a.fx_curve[0]),
        "모델 주당가치(8.4%)": float(r["per_share"][0]),
        "사이클평균 마진": cycle,
        "모델(사이클평균)": float(r_cycle["per_share"][0]),
        "괴리(사이클평균)": float(r_cycle["per_share"][0]) / price - 1,
        "모델(마진수렴)": float(r_fade["per_share"][0]),
        "괴리(마진수렴)": float(r_fade["per_share"][0]) / price - 1,
        "정상화 매출(억원)": norm_rev,
        "모델(매출+마진 정상화)": float(r_norm["per_share"][0]),
        "괴리(매출+마진)": float(r_norm["per_share"][0]) / price - 1,
        "내재 WACC(기준연도)": implied_wacc["기준연도"],
        "내재 WACC(사이클평균)": implied_wacc["사이클평균"],
        "내재 WACC(마진수렴)": implied_wacc["마진수렴"],
        "내재 WACC(매출+마진)": implied_wacc["매출+마진"],
        "무위험이자율": m.risk_free,
        "회사채AA-": m.corp_aa,
        "신용스프레드(회사채-국고채)": m.corp_aa - m.risk_free,
        "재레버리지 베타": w.relevered_beta,
        "내재 자기자본비용(마진수렴)": implied_ke,
        "내재 ERP(마진수렴)": implied_erp,
        "모델 주당가치(CAPM)": float(r_capm["per_share"][0]) if r_capm is not None else float("nan"),
        "CAPM WACC": w.capm_wacc,
        "실제 주가": price,
        "괴리(8.4%)": float(r["per_share"][0]) / price - 1,
        "1년 후 주가": price_1y,
        "1년 주가 변화": price_1y / price - 1,
        "이후 1년 평균 유가": _forward_oil(c, m.valuation_date),
        "모델 EV/EBITDA": float(r["ev"][0]) / ebitda,
        "시장 EV/EBITDA": (m.market_cap + float(r["net_debt"][0]) + a.nci) / ebitda,
    }


def main():
    load_env()
    cfg = load_config()
    rows = []
    for base_year, (val_date, usd) in CASES.items():
        print(f"\n===== 기준 {base_year}년 · 평가일 {val_date} =====")
        rows.append(run_case(cfg, base_year, val_date, usd))
    df = pd.DataFrame(rows)
    out = ROOT / "output" / "backtest.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    show = df.copy()
    for c in ("EBITDA 마진", "비유류비/매출", "사이클평균 마진", "괴리(8.4%)", "괴리(사이클평균)",
              "괴리(마진수렴)", "괴리(매출+마진)", "1년 주가 변화", "CAPM WACC",
              "내재 WACC(기준연도)", "내재 WACC(사이클평균)", "내재 WACC(마진수렴)", "내재 WACC(매출+마진)",
              "무위험이자율", "회사채AA-", "신용스프레드(회사채-국고채)", "내재 자기자본비용(마진수렴)", "내재 ERP(마진수렴)"):
        show[c] = show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for c in ("EBITDA(억원)", "정상화 매출(억원)", "모델 주당가치(8.4%)", "모델(사이클평균)", "모델(마진수렴)",
              "모델(매출+마진 정상화)", "모델 주당가치(CAPM)", "실제 주가", "1년 후 주가"):
        show[c] = show[c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
    for c in ("기준 유가", "이후 1년 평균 유가", "기준 환율", "모델 EV/EBITDA", "시장 EV/EBITDA"):
        show[c] = show[c].map(lambda v: f"{v:,.1f}")
    pd.set_option("display.width", 260)
    cols = ["평가일", "기준연도", "실제 주가", "모델 주당가치(8.4%)", "괴리(8.4%)",
            "모델(사이클평균)", "괴리(사이클평균)", "모델(마진수렴)", "괴리(마진수렴)",
            "모델(매출+마진 정상화)", "괴리(매출+마진)"]
    print("\n" + show[cols].to_string(index=False))
    risk_cols = ["평가일", "기준연도", "무위험이자율", "신용스프레드(회사채-국고채)", "재레버리지 베타",
                 "내재 WACC(기준연도)", "내재 WACC(사이클평균)", "내재 WACC(마진수렴)", "내재 WACC(매출+마진)",
                 "내재 자기자본비용(마진수렴)", "내재 ERP(마진수렴)"]
    show["재레버리지 베타"] = show["재레버리지 베타"].map(lambda v: f"{v:.3f}")
    print("\n[시기별 시장위험 역산]\n" + show[risk_cols].to_string(index=False))
    err = df[["괴리(8.4%)", "괴리(사이클평균)", "괴리(마진수렴)", "괴리(매출+마진)"]].abs()
    print("\n평균 절대 괴리: " + " | ".join(f"{c} {err[c].mean():.0%}" for c in err.columns))
    chart = ROOT / "output" / "charts" / "backtest.png"
    report.chart_backtest(df, chart)
    report.chart_implied_risk(df, ROOT / "output" / "charts" / "implied_risk.png")
    print(f"\n저장: {out}, {chart}")


if __name__ == "__main__":
    main()
