"""실행: python run.py          (실제 데이터, .env 에 API 키 필요)
       python run.py --demo   (가상 데이터로 전체 흐름 확인)
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd

from src import demo, report
from src.config import ROOT, load_config, load_env
from src.excel_check import check
from src.market import FlatCurve
from src.model import build_assumptions, compute_wacc, dcf
from src.implied import solve_all, table as implied_table
from src.risk import grid_oil_fx, grid_wacc_growth, hedge_analysis, simulate_paths


def main():
    ap = argparse.ArgumentParser(description="유가·환율 연동 DCF 밸류에이션")
    ap.add_argument("--demo", action="store_true", help="API 키 없이 가상 데이터로 실행")
    ap.add_argument("--refresh", action="store_true", help="캐시를 무시하고 API 를 다시 호출")
    ap.add_argument("--skip-excel-check", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config()
    latest = None
    if args.demo:
        cfg = copy.deepcopy(cfg)
        cfg["company"]["name"] = demo.DEMO_NAME
        cfg["exposure"].update(usd_revenue_share=0.40, usd_nonfuel_cost_share=0.15, fuel_pass_through=0.5)
        cfg["wacc"]["beta_override"] = 0.0
        hist, mapping = demo.demo_history(), pd.DataFrame()
        market = demo.demo_market(cfg["company"]["valuation_date"])
        out = ROOT / "output" / "demo"
    else:
        from src.pipeline import load_financials, load_market
        print("[1/4] DART 재무제표 수집")
        hist, mapping, latest = load_financials(cfg, args.refresh)
        print("[2/4] ECOS·주식시세 수집")
        market = load_market(cfg, args.refresh)
        out = ROOT / "output"

    print("[3/4] 가치평가")
    shares = cfg["overrides"]["shares"].get("shares_outstanding") or market.shares
    last = hist.iloc[-1]
    debt_total = latest["debt_total"] if latest else float(last["debt"] + last["lease_liabilities"])
    w = compute_wacc(cfg, market, debt_total)
    curve = FlatCurve(market.oil_base, market.fx_base)
    a = build_assumptions(cfg, hist, market, curve, shares)
    if latest:
        a.net_debt, a.nci = latest["net_debt"], latest["nci"]
        print(f"  순차입금: {latest['label']} 기준")
    # 순차입금이 환산된 시점의 환율 (F 경로에서 평가일 현물로 재환산할 때의 기준)
    bs_date = latest["date"] if latest else pd.Timestamp(a.base_year, 12, 31)
    a.fx_at_balance_sheet = float(market.fx_daily[market.fx_daily.index <= bs_date].iloc[-1])
    if a.usd_net_debt:
        adj = a.usd_net_debt * (a.fx_spot - a.fx_at_balance_sheet) / 1e8
        print(f"  환율 재평가: 순외화부채 {a.usd_net_debt/1e8:.0f}억달러, "
              f"{bs_date.date()} {a.fx_at_balance_sheet:,.1f}원 → 평가일 {a.fx_spot:,.1f}원 = 순차입금 {adj:+,.0f}억원")

    if w.override is not None:
        print(f"  할인율: 채택 {w.wacc:.2%} (공시값) / CAPM 계산값 {w.capm_wacc:.2%} — {w.beta_source}")
    base = dcf(a, w.wacc)
    sc, sim = cfg["scenario"], cfg["simulation"]

    # 시장가격 역산과 가정 조합 비교
    base_ebitda = float(hist["ebitda"].iloc[-1])
    imp = solve_all(a, w.wacc, market.market_cap, base_ebitda)
    implied_rows = pd.DataFrame(implied_table(imp, a, w.wacc))
    cases = []
    for k, klabel in ((cfg["forecast"]["ratio_years"], f"{a.base_year}년만" if cfg["forecast"]["ratio_years"] == 1
                       else f"최근 {cfg['forecast']['ratio_years']}년 평균"), (3, "2022~2024 평균")):
        c = copy.deepcopy(cfg); c["forecast"]["ratio_years"] = k
        ak = build_assumptions(c, hist, market, curve, shares)
        ak.net_debt, ak.nci = a.net_debt, a.nci
        ak.fx_at_balance_sheet = a.fx_at_balance_sheet
        for wlabel, wv in (("CAPM 계산값", w.capm_wacc), ("채택(공시 8.4%)", w.wacc)):
            r = dcf(ak, wv)
            cases.append({"비율 기간": klabel, "비유류비/매출": f"{ak.nonfuel_ratio:.1%}", "할인율": f"{wlabel} {wv:.2%}",
                          "EV(억원)": float(r["ev"][0]), "EV/EBITDA": float(r["ev"][0]) / base_ebitda,
                          "주당가치(원)": float(r["per_share"][0])})
    cases_df = pd.DataFrame(cases).drop_duplicates(subset=["비율 기간", "할인율"])
    sens_wg = grid_wacc_growth(a, w.wacc, sc["wacc_steps"], sc["growth_steps"])
    sens_of = grid_oil_fx(a, w.wacc, sc["oil_shocks"], sc["fx_shocks"])
    oil_p, fx_p = simulate_paths(a, market.oil_vol, market.fx_vol, market.oil_fx_corr, sim["paths"], sim["seed"])
    hedge, samples, hedge_ebitda, ebitda_label = hedge_analysis(a, w.wacc, oil_p, fx_p, sim["hedge_ratios"], sim["hedge_years"])
    exit_vals = [float(dcf(a, w.wacc, exit_multiple=m)["per_share"][0]) for m in sc["exit_ev_ebitda"]]
    mc0 = samples[sim["hedge_ratios"][0]]

    zero_oil = f"유가 {0:+.0%}"
    zero_fx = f"환율 {0:+.0%}"
    ranges = pd.DataFrame({
        "low": [sens_wg.values.min(), sens_of[zero_fx].min(), sens_of.loc[zero_oil].min(),
                np.percentile(mc0, 10), min(exit_vals)],
        "high": [sens_wg.values.max(), sens_of[zero_fx].max(), sens_of.loc[zero_oil].max(),
                 np.percentile(mc0, 90), max(exit_vals)],
    }, index=[
        f"DCF: WACC ±{max(sc['wacc_steps']):.1%}, g ±{max(sc['growth_steps']):.2%}",
        f"유가 {min(sc['oil_shocks']):+.0%} ~ {max(sc['oil_shocks']):+.0%}",
        f"환율 {min(sc['fx_shocks']):+.0%} ~ {max(sc['fx_shocks']):+.0%}",
        "몬테카를로 P10 ~ P90 (무헤지)",
        f"EV/EBITDA {min(sc['exit_ev_ebitda']):.1f}x ~ {max(sc['exit_ev_ebitda']):.1f}x",
    ])
    ranges = ranges.clip(lower=0.0)   # 주주 유한책임: 음수 주주가치는 0원
    price = float(market.stock_daily[market.stock_daily.index <= market.valuation_date].iloc[-1]) \
        if market.stock_daily is not None else None

    print("[4/4] 결과물 작성")
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    per_share = float(base["per_share"][0])
    report.chart_football_field(ranges, per_share, price, charts / "football_field.png", args.demo)
    report.chart_cases(cases_df, price or per_share, charts / "cases.png", args.demo)
    report.chart_heatmap(sens_of, per_share, "유가 × 환율 민감도 (주당가치, 원)", charts / "sensitivity_oil_fx.png", args.demo)
    first_full = a.years.index(int(ebitda_label.split()[0]))
    report.chart_hedge(hedge_ebitda, ebitda_label, float(base["projection"]["ebitda"][0, first_full]),
                       charts / "hedge_ebitda.png", args.demo)
    oil_hist = market.oil_monthly.loc[market.valuation_date - pd.DateOffset(years=10):]
    report.chart_market(oil_hist, market.fx_daily, market.oil_base, market.fx_base, charts / "market_history.png", args.demo)

    summary = {"WACC": w.wacc, "기업가치(억원)": float(base["ev"][0]), "주당가치(원)": per_share,
               "시장 주당가격(원)": price or 0.0, "내재 WACC": imp.wacc or 0.0}
    xlsx = out / "valuation_model.xlsx"
    cells = report.write_excel(xlsx, a, w, hist, mapping, sens_wg, sens_of, hedge, summary, args.demo,
                               extra={"시장역산": implied_rows, "가정조합비교": cases_df.round(1)})
    excel_status = "생략 (--skip-excel-check)" if args.skip_excel_check else check(xlsx, {
        f"DCF!{cells['per_share_cell'].split('!')[1]}": per_share,
        f"DCF!{cells['ev_cell'].split('!')[1]}": float(base["ev"][0]),
        cells["wacc_cell"]: w.wacc,
    })
    cases_md = cases_df.copy()
    cases_md["EV(억원)"] = cases_md["EV(억원)"].map(lambda v: f"{v:,.0f}")
    cases_md["EV/EBITDA"] = cases_md["EV/EBITDA"].map(lambda v: f"{v:.1f}배")
    cases_md["주당가치(원)"] = cases_md["주당가치(원)"].map(lambda v: f"{v:,.0f}")
    report.write_summary(out / "summary.md", a, w, base, ranges, sens_of, hedge, price, args.demo, excel_status,
                         implied=implied_rows, cases=cases_md)

    print()
    print(f"  WACC {w.wacc:.2%} | EV {base['ev'][0]:,.0f}억원 | 주당가치 {per_share:,.0f}원"
          + (f" | 현재가 {price:,.0f}원" if price else ""))
    print(f"  엑셀 검증: {excel_status}")
    print(f"  결과물: {out}")


if __name__ == "__main__":
    main()
