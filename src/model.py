"""WACC, 유가·환율 연동 추정 손익, FCFF 기반 DCF.

금액 단위는 억원. 유가는 달러/배럴, 환율은 원/달러.
모든 계산은 (시나리오 수 P) × (예측 연도 N) 배열로 처리해서 몬테카를로에도 그대로 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .market import MarketInputs, blume_adjust

WON_PER_EOK = 1e8  # 1억원 = 1e8 원


@dataclass
class Wacc:
    risk_free: float
    raw_beta: float
    adjusted_beta: float
    unlevered_beta: float
    relevered_beta: float
    erp: float
    size_premium: float
    cost_of_equity: float
    pre_tax_kd: float
    after_tax_kd: float
    equity_value: float
    debt_value: float
    equity_weight: float
    debt_weight: float
    tax_rate: float
    beta_source: str
    override: float | None = None      # 채택 할인율(공시값 등). 있으면 CAPM 계산값 대신 이 값을 쓴다
    capm_wacc: float = field(init=False)
    wacc: float = field(init=False)

    def __post_init__(self):
        self.capm_wacc = self.equity_weight * self.cost_of_equity + self.debt_weight * self.after_tax_kd
        self.wacc = self.capm_wacc if self.override is None else self.override


def compute_wacc(cfg: dict, market: MarketInputs, debt_value: float) -> Wacc:
    w = cfg["wacc"]
    t = w["tax_rate"]
    if w.get("beta_override"):
        raw, source = float(w["beta_override"]), "config 입력값"
    elif market.raw_beta is not None:
        raw = market.raw_beta
        source = (f"{market.beta_index_name} 대비 주간수익률 회귀 "
                  f"({cfg['market_data']['beta_lookback_weeks']}주, R² {market.beta_r2:.1%})")
    else:
        raise ValueError("베타를 계산할 주가 데이터가 없습니다. config.toml 의 wacc.beta_override 를 입력하세요.")
    if market.market_cap is None:
        raise ValueError("시가총액이 없습니다. 공공데이터포털 주식시세 API 키를 확인하세요.")
    adj = blume_adjust(raw)
    de_now = debt_value / market.market_cap
    unlev = adj / (1 + (1 - t) * de_now)
    de_target = w.get("target_debt_to_equity") or de_now
    relev = unlev * (1 + (1 - t) * de_target)
    e_w = 1 / (1 + de_target)
    kd = market.corp_aa + w["credit_spread_over_aa"]
    return Wacc(
        risk_free=market.risk_free, raw_beta=raw, adjusted_beta=adj, unlevered_beta=unlev, relevered_beta=relev,
        erp=w["equity_risk_premium"], size_premium=w["size_premium"],
        cost_of_equity=market.risk_free + relev * w["equity_risk_premium"] + w["size_premium"],
        pre_tax_kd=kd, after_tax_kd=kd * (1 - t), equity_value=market.market_cap, debt_value=debt_value,
        equity_weight=e_w, debt_weight=1 - e_w, tax_rate=t, beta_source=source,
        override=(float(w["wacc_override"]) if w.get("wacc_override") else None),
    )


@dataclass
class Assumptions:
    company: str
    base_year: int
    valuation_date: pd.Timestamp
    years: list[int]
    growth: np.ndarray
    terminal_growth: float
    mid_year: bool
    tax_rate: float
    revenue_last: float
    fuel_volume_last: float        # 유가 환산 소비량 (배럴 상당)
    fx_ref: float                  # 기준연도 평균 환율 (환율 효과의 기준점)
    oil_ref: float                 # 기준연도 평균 유가 (유류할증 전가의 기준점)
    fuel_pass_through: float       # 유가 변동분 중 매출로 전가되는 비율
    usd_revenue_share: float
    usd_nonfuel_cost_share: float
    nonfuel_ratio: float | np.ndarray   # 연도별 배열을 주면 마진이 해마다 달라진다(정상화 수렴)
    da_ratio: float
    capex_ratio: float
    capex_floor_to_da: float
    nwc_ratio: float
    nwc_last: float
    terminal_capex_to_da: float
    net_debt: float                # 재무상태표 공시일 기준 순차입금 (그날 환율로 환산된 금액)
    nci: float
    shares: float
    oil_curve: np.ndarray
    fx_curve: np.ndarray
    curve_source: str
    # F 경로: 달러 순부채의 환율 재평가
    usd_net_debt: float = 0.0      # 달러 단위 순외화부채 (사업보고서 시장위험 공시)
    fx_at_balance_sheet: float = 0.0   # 순차입금이 환산된 시점의 환율
    fx_spot: float = 0.0           # 평가일 현물환율


def _ratio(hist: pd.DataFrame, num: str, n: int, override: float) -> float:
    if override:
        return float(override)
    recent = hist.tail(n)
    return float((recent[num] / recent["revenue"]).mean())


def build_assumptions(cfg: dict, hist: pd.DataFrame, market: MarketInputs, curve, shares: float) -> Assumptions:
    if "fuel_cost" not in hist.columns or pd.isna(hist["fuel_cost"].iloc[-1]):
        raise ValueError("기준연도 유류비가 없습니다. 사업보고서 영업비용 주석의 유류비를 config.toml [overrides.fuel_cost] 에 입력하세요.")
    f = cfg["forecast"]
    base = int(hist.index[-1])
    n = f["years"]
    years = list(range(base + 1, base + 1 + n))
    growth = np.asarray(f["revenue_growth"], dtype=float)
    if len(growth) != n:
        raise ValueError("forecast.revenue_growth 길이가 forecast.years 와 다릅니다.")
    k = f.get("ratio_years", 3)
    fx_ref = market.fx_annual_avg(base)
    oil_ref = market.oil_annual_avg(base)
    fuel_hist = hist.dropna(subset=["fuel_cost"]).tail(k)
    return Assumptions(
        company=cfg["company"]["name"], base_year=base, valuation_date=market.valuation_date, years=years,
        growth=growth, terminal_growth=f["terminal_growth"], mid_year=f["mid_year_convention"],
        tax_rate=cfg["wacc"]["tax_rate"],
        revenue_last=float(hist["revenue"].iloc[-1]),
        fuel_volume_last=float(hist["fuel_cost"].iloc[-1]) * WON_PER_EOK / (oil_ref * fx_ref),
        fx_ref=fx_ref, oil_ref=oil_ref,
        fuel_pass_through=cfg["exposure"]["fuel_pass_through"],
        usd_revenue_share=cfg["exposure"]["usd_revenue_share"],
        usd_nonfuel_cost_share=cfg["exposure"]["usd_nonfuel_cost_share"],
        nonfuel_ratio=float((fuel_hist["nonfuel_cash_opex"] / fuel_hist["revenue"]).mean()),
        da_ratio=_ratio(hist, "depreciation", k, f["da_to_revenue"]),
        capex_ratio=_ratio(hist, "capex", k, f["capex_to_revenue"]),
        capex_floor_to_da=f["capex_floor_to_da"],
        nwc_ratio=_ratio(hist, "operating_nwc", k, f["nwc_to_revenue"]),
        nwc_last=float(hist["operating_nwc"].iloc[-1]),
        terminal_capex_to_da=f["terminal_capex_to_da"],
        net_debt=float(hist["net_debt"].iloc[-1]), nci=float(hist["nci"].iloc[-1]), shares=shares,
        oil_curve=curve.oil_curve(years), fx_curve=curve.fx_curve(years), curve_source=curve.source,
        usd_net_debt=float(cfg["exposure"].get("usd_net_debt", 0.0)),
        fx_at_balance_sheet=fx_ref,      # 기본값. 최신 재무상태표를 쓰면 run.py 에서 그 시점 환율로 바꾼다
        fx_spot=market.fx_spot,
    )


def timing(a: Assumptions) -> tuple[np.ndarray, np.ndarray, float]:
    """(연도별 반영비율, 연도별 할인기간, 마지막 연도 말까지 기간). 평가일 이후 남은 기간만 반영한다."""
    ends = np.array([(pd.Timestamp(y, 12, 31) - a.valuation_date).days / 365.25 for y in a.years])
    frac = np.clip(ends, 0.0, 1.0)
    t = ends - frac / 2 if a.mid_year else ends
    return frac, t, float(ends[-1])


def project(a: Assumptions, oil: np.ndarray | None = None, fx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """연도별 추정 손익과 FCFF. oil, fx: (P, N) 또는 (N,) 배열. 없으면 기준 커브."""
    oil = np.atleast_2d(a.oil_curve if oil is None else oil)
    fx = np.atleast_2d(a.fx_curve if fx is None else fx)
    cum = np.cumprod(1 + a.growth)
    real_rev = a.revenue_last * cum
    fx_chg = fx / a.fx_ref - 1
    fuel_volume = a.fuel_volume_last * cum
    fuel = fuel_volume * oil * fx / WON_PER_EOK
    surcharge = a.fuel_pass_through * fuel_volume * (oil - a.oil_ref) * fx / WON_PER_EOK
    revenue = real_rev * (1 + a.usd_revenue_share * fx_chg) + surcharge
    nonfuel = a.nonfuel_ratio * real_rev * (1 + a.usd_nonfuel_cost_share * fx_chg)
    ebitda = revenue - fuel - nonfuel
    da = a.da_ratio * revenue
    ebit = ebitda - da
    tax = a.tax_rate * np.maximum(ebit, 0.0)
    nopat = ebit - tax
    capex = np.maximum(a.capex_ratio * revenue, a.capex_floor_to_da * da)
    nwc = a.nwc_ratio * revenue
    prev = np.concatenate([np.full((nwc.shape[0], 1), a.nwc_last), nwc[:, :-1]], axis=1)
    d_nwc = nwc - prev
    fcff = nopat + da - capex - d_nwc
    return {"revenue": revenue, "fuel_cost": fuel, "nonfuel_opex": nonfuel, "ebitda": ebitda, "da": da,
            "ebit": ebit, "tax": tax, "nopat": nopat, "capex": capex, "nwc": nwc, "d_nwc": d_nwc, "fcff": fcff,
            "oil": np.broadcast_to(oil, fcff.shape), "fx": np.broadcast_to(fx, fcff.shape),
            "fuel_volume": np.broadcast_to(fuel_volume, fcff.shape)}


def terminal_fcff(a: Assumptions, p: dict[str, np.ndarray], g: float) -> np.ndarray:
    """영구기간 첫해 FCFF: 마지막 예측연도 마진을 유지하고 CAPEX 는 감가상각비 대비 정상화."""
    rev_n = p["revenue"][:, -1]
    rev_t = rev_n * (1 + g)
    ebitda_t = p["ebitda"][:, -1] / rev_n * rev_t
    da_t = a.da_ratio * rev_t
    ebit_t = ebitda_t - da_t
    nopat_t = ebit_t - a.tax_rate * np.maximum(ebit_t, 0.0)
    capex_t = a.terminal_capex_to_da * da_t
    d_nwc_t = a.nwc_ratio * (rev_t - rev_n)
    return nopat_t + da_t - capex_t - d_nwc_t


def dcf(a: Assumptions, wacc: float, g: float | None = None, oil=None, fx=None,
        exit_multiple: float | None = None) -> dict[str, np.ndarray]:
    """기업가치(EV), 주주가치, 주당가치. exit_multiple 을 주면 영구성장 대신 EV/EBITDA 배수로 잔존가치 계산."""
    g = a.terminal_growth if g is None else g
    if exit_multiple is None and wacc <= g:
        raise ValueError("WACC 가 영구성장률보다 커야 합니다.")
    p = project(a, oil, fx)
    frac, t, t_end = timing(a)
    disc = (1 + wacc) ** -t
    pv_fcff = (p["fcff"] * frac * disc).sum(axis=1)
    if exit_multiple is None:
        tv = terminal_fcff(a, p, g) / (wacc - g)
        tv_disc = (1 + wacc) ** -(t_end - 0.5 if a.mid_year else t_end)
    else:
        tv = p["ebitda"][:, -1] * exit_multiple
        tv_disc = (1 + wacc) ** -t_end
    pv_tv = tv * tv_disc
    ev = pv_fcff + pv_tv
    net_debt = fx_adjusted_net_debt(a, p["fx"][:, 0])
    equity = ev - net_debt - a.nci
    return {"ev": ev, "pv_fcff": pv_fcff, "pv_tv": pv_tv, "tv": tv, "equity": equity, "net_debt": net_debt,
            "per_share": equity * WON_PER_EOK / a.shares, "projection": p, "frac": frac, "t": t}


def fx_adjusted_net_debt(a: Assumptions, fx_year1: np.ndarray) -> np.ndarray:
    """달러 순부채를 평가 시점 환율로 다시 환산한 순차입금 (F 경로).

    재무상태표의 순차입금은 공시일 환율로 환산돼 있다. 평가일 현물환율로 바꾸고,
    환율 시나리오에서는 1차연도 환율이 움직인 비율만큼 현물환율도 같이 움직인다고 본다.
    """
    if not a.usd_net_debt:
        return np.full(np.shape(fx_year1), a.net_debt, dtype=float)
    shock = np.asarray(fx_year1, dtype=float) / a.fx_curve[0]
    fx_used = a.fx_spot * shock
    return a.net_debt + a.usd_net_debt * (fx_used - a.fx_at_balance_sheet) / WON_PER_EOK
