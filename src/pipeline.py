"""API 에서 데이터를 모아 재무 이력과 시장 입력값을 만든다."""
from __future__ import annotations

import pandas as pd

from .config import api_key
from .financials import build_history, extract_year
from . import market_index
from .market import MarketInputs, last_on_or_before, regression_beta, vol_and_corr
from .sources import dart, datagokr, ecos


# 보고서 코드 → 재무상태표 기준일 (순차입금이 환산된 시점)
REPORT_END = {"11013": (3, 31), "11012": (6, 30), "11014": (9, 30), "11011": (12, 31)}


def _need(name: str) -> str:
    key = api_key(name)
    if not key:
        raise SystemExit(f"{name} 가 없습니다. .env.example 을 .env 로 복사하고 키를 입력하세요.")
    return key


def load_financials(cfg: dict, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    key = _need("DART_API_KEY")
    c = cfg["company"]
    corp, corp_name = dart.corp_code(key, c["stock_code"], refresh)
    print(f"  DART 고유번호 {corp} ({corp_name})")
    reports = {y: dart.financial_statements(key, corp, y, c["fs_div"], refresh) for y in c["history_years"]}
    ov = cfg.get("overrides", {})
    pf = {k: v for k, v in cfg.get("proforma", {}).items() if k.isdigit()}
    hist, log = build_history(reports, {k: ov.get(k, {}) for k in ("fuel_cost", "depreciation")}, pf)
    check = cfg.get("proforma", {}).get("check_revenue")
    if check and int(check["year"]) in hist.index:
        got = float(hist.loc[int(check["year"]), "revenue"])
        gap = got - float(check["amount"])
        mark = "일치" if abs(gap) < 1.0 else f"차이 {gap:+,.0f}억원"
        print(f"  pro forma 매출 대사({check['year']}): 계산 {got:,.0f} vs 공시 {float(check['amount']):,.0f} → {mark}")

    latest = None
    bs = c.get("latest_balance_sheet")
    if bs:  # 기준연도 이후 분기·반기 보고서의 재무상태표로 순차입금을 갱신
        rows = dart.financial_statements(key, corp, bs["year"], c["fs_div"], refresh, report_code=bs["report_code"])
        vals = extract_year(rows, bs["year"]).values
        latest = {
            "debt_total": vals.get("debt", 0.0) + vals.get("lease_liabilities", 0.0),
            "net_debt": vals.get("debt", 0.0) + vals.get("lease_liabilities", 0.0) - vals["cash"],
            "nci": vals.get("nci", 0.0),
            "label": f"{bs['year']}년 보고서({bs['report_code']})",
            "date": pd.Timestamp(bs["year"], *REPORT_END[bs["report_code"]]),
        }
    return hist, log, latest


def load_market(cfg: dict, refresh: bool = False) -> MarketInputs:
    key = _need("ECOS_API_KEY")
    md = cfg["market_data"]
    val = pd.Timestamp(cfg["company"]["valuation_date"])
    first_year = min(cfg["company"]["history_years"])
    start = min(val - pd.DateOffset(months=md["vol_lookback_months"] + 2), pd.Timestamp(first_year, 1, 1))
    d0, d1 = start.strftime("%Y%m%d"), val.strftime("%Y%m%d")

    ktb = ecos.get_series(key, "ktb10y", d0, d1, refresh)
    corp = ecos.get_series(key, "corp_aa3y", d0, d1, refresh)
    fx = ecos.get_series(key, "usdkrw", d0, d1, refresh)
    kospi = ecos.get_series(key, "kospi", d0, d1, refresh)
    oil = ecos.get_oil_monthly(key, md["oil_benchmark"], start.strftime("%Y%m"), val.strftime("%Y%m"), refresh)

    lookback = val - pd.DateOffset(months=md["oil_lookback_months"])
    oil_recent = oil[(oil.index > lookback) & (oil.index <= val)]
    fx_recent = fx[(fx.index > lookback) & (fx.index <= val)]
    oil_vol, fx_vol, corr = vol_and_corr(oil, fx, md["vol_lookback_months"])

    # 시가총액(WACC 가중치)과 주가(베타)는 공공데이터포털 주식시세 API 에서 가져온다
    dg_key = _need("DATA_GO_KR_API_KEY")
    s0 = (val - pd.Timedelta(weeks=md["beta_lookback_weeks"] + 6)).strftime("%Y%m%d")
    px = datagokr.stock_prices(dg_key, cfg["company"]["stock_code"], s0, d1, refresh)
    stock = px["close"]
    last = px[px.index <= val].iloc[-1]
    market_cap, shares = float(last["market_cap"]), float(last["shares"])

    # 베타 추정용 시장지수 선택
    index_name, index_series, semi_w = "KOSPI", kospi, 0.0
    if md.get("market_index", "KOSPI").upper() == "KOSPI_EX_SEMIS":
        mcap = ecos.get_series(key, "kospi_mcap", d0, d1, refresh)          # 유가증권시장 시가총액(억원)
        semi_px = {c: datagokr.stock_prices(dg_key, c, s0, d1, refresh) for c in market_index.SEMIS}
        index_series, w = market_index.ex_semis_index(
            kospi, mcap,
            {c: d["close"] for c, d in semi_px.items()},
            {c: d["market_cap"] for c, d in semi_px.items()},
        )
        index_name, semi_w = "KOSPI 반도체 제외", float(w[w.index <= val].iloc[-1])
        print(f"  베타 시장지수: {index_name} (평가일 반도체 비중 {semi_w:.1%})")
    raw_beta, r2, n_obs = regression_beta(stock, index_series, md["beta_lookback_weeks"])
    print(f"  원시 베타 {raw_beta:.3f} (R² {r2:.1%}, 주간 관측치 {n_obs}개)")

    return MarketInputs(
        valuation_date=val,
        risk_free=last_on_or_before(ktb, val) / 100,
        corp_aa=last_on_or_before(corp, val) / 100,
        fx_spot=last_on_or_before(fx, val),
        fx_base=float(fx_recent.mean()),
        oil_base=float(oil_recent.mean()),
        oil_vol=oil_vol, fx_vol=fx_vol, oil_fx_corr=corr,
        raw_beta=raw_beta, market_cap=market_cap, shares=shares,
        beta_index_name=index_name, beta_index=index_series, semi_weight=semi_w, beta_r2=r2,
        oil_monthly=oil, fx_daily=fx, kospi_daily=kospi, stock_daily=stock,
    )
