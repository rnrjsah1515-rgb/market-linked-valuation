"""분기 데이터셋 만들기 (2019Q1~2026Q2).

- 매출·영업이익: DART 재무제표 API (분기 3개월 금액)
- 유류비: 정기보고서 '비용의 성격별 분류' 주석 (연결, 3개월) — src/quarterly.py
- 유가(두바이)·원/달러: ECOS 분기 평균
- 검증: 1~4분기 유류비 합계를 연간 공시액과 대조

실행: python build_quarterly.py  →  output/quarterly.csv
"""
from __future__ import annotations

import pandas as pd

from src.config import ROOT, api_key, load_config, load_env
from src.jetfuel import quarterly_usd_per_barrel
from src.quarterly import expense_note, report_text
from src.sources import dart, ecos, http

# 분기·반기 보고서 (4분기는 연간에서 차감해 계산)
REPORTS = {
    "2019Q1": "20190515002727", "2019Q2": "20190814002310", "2019Q3": "20191114001886",
    "2020Q1": "20200515001367", "2020Q2": "20200814001952", "2020Q3": "20201116001718",
    "2021Q1": "20210514000933", "2021Q2": "20210813001202", "2021Q3": "20211112000693",
    "2022Q1": "20220513001160", "2022Q2": "20220812001185", "2022Q3": "20221111000882",
    "2023Q1": "20230512000925", "2023Q2": "20230814002772", "2023Q3": "20231114001680",
    "2024Q1": "20240514001467", "2024Q2": "20240814004021", "2024Q3": "20241114002512",
    "2025Q1": "20250515002853", "2025Q2": "20250814004301", "2025Q3": "20251114002585",
    "2026Q1": "20260515001941", "2026Q2": "20260814002803",
}
# 연간 연결 유류비(억원) — 각 사업보고서 주석. 4분기 계산과 검증에 쓴다.
ANNUAL_FUEL = {2019: 31716.44, 2020: 12382.34, 2021: 17860.81, 2022: 41362.38,
               2023: 48023.39, 2024: 49808.48, 2025: 67419.62}
QUARTER_REPORT_CODE = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}


def quarterly_income(key: str, corp: str, years: range) -> pd.DataFrame:
    """분기별 매출·영업이익(3개월, 억원). API 의 thstrm_amount 는 해당 분기 3개월 금액이다."""
    rows = []
    for year in years:
        for q, code in QUARTER_REPORT_CODE.items():
            try:
                data = dart.financial_statements(key, corp, year, "CFS", report_code=code)
            except Exception as e:                      # 아직 공시되지 않은 분기
                print(f"  {year}Q{q}: {str(e)[:60]}")
                continue
            vals = {}
            for r in data:
                if r["sj_div"] not in ("IS", "CIS"):
                    continue
                name = r["account_nm"].strip()
                amount = (r.get("thstrm_amount") or "").replace(",", "")
                if not amount.lstrip("-").isdigit():
                    continue
                if name in ("매출액", "영업수익", "수익(매출액)", "매출"):
                    vals.setdefault("revenue", int(amount) / 1e8)
                elif name.startswith("영업이익"):
                    vals.setdefault("operating_income", int(amount) / 1e8)
            if "revenue" in vals:
                rows.append({"quarter": f"{year}Q{q}", "year": year, "q": q, **vals})
    df = pd.DataFrame(rows).set_index("quarter")
    # 사업보고서(4분기 자리)의 금액은 '연간' 이다. 4분기 = 연간 − 1~3분기 합.
    for year in df["year"].unique():
        q4 = f"{year}Q4"
        q123 = [f"{year}Q{i}" for i in (1, 2, 3)]
        if q4 in df.index and all(q in df.index for q in q123):
            for col in ("revenue", "operating_income"):
                df.loc[q4, col] = df.loc[q4, col] - df.loc[q123, col].sum()
        elif q4 in df.index:
            df = df.drop(index=q4)          # 1~3분기가 없으면 4분기를 만들 수 없다
    return df.drop(columns=["year", "q"])


def quarterly_market(key: str, start: str, end: str) -> pd.DataFrame:
    """분기 평균 두바이유(달러/배럴)와 원/달러 환율."""
    oil = ecos.get_oil_monthly(key, "010102", start[:6], end[:6])
    fx = ecos.get_series(key, "usdkrw", start, end)
    oil_q = oil.resample("QE").mean()
    fx_q = fx.resample("QE").mean()
    df = pd.concat([oil_q.rename("oil"), fx_q.rename("fx")], axis=1).dropna()
    df.index = [f"{d.year}Q{d.quarter}" for d in df.index]
    df.index.name = "quarter"
    return df


def main():
    load_env()
    cfg = load_config()
    dart_key, ecos_key = api_key("DART_API_KEY"), api_key("ECOS_API_KEY")
    corp, _ = dart.corp_code(dart_key, cfg["company"]["stock_code"])

    print("[1/3] 분기 유류비 (정기보고서 주석)")
    fuel = {}
    for label, rcept in sorted(REPORTS.items()):
        found = expense_note(report_text(dart_key, rcept))
        if found is None:
            print(f"  {label}: 연결 표를 찾지 못함")
            continue
        fuel[label] = found[0]
        print(f"  {label}: {found[0]:>9,.0f}억원")
    for year, total in ANNUAL_FUEL.items():
        q123 = [fuel.get(f"{year}Q{q}") for q in (1, 2, 3)]
        if all(v is not None for v in q123):
            fuel[f"{year}Q4"] = total - sum(q123)

    print("\n[2/3] 분기 손익 (DART 재무제표 API)")
    income = quarterly_income(dart_key, corp, range(2019, 2027))

    print("\n[3/3] 분기 유가·환율 (ECOS)")
    market = quarterly_market(ecos_key, "20190101", "20260930")

    jet = quarterly_usd_per_barrel()                       # 분기 평균 현물
    jet_lag = quarterly_usd_per_barrel(lag_weeks=4).rename("jet_lagged")   # 매입·소비 시차 4주
    df = (income.join(pd.Series(fuel, name="fuel_cost")).join(market)
          .join(jet).join(jet_lag).sort_index())
    df["crack_ratio"] = df["jet"] / df["oil"]                     # 항공유 / 원유
    df["fuel_share"] = df["fuel_cost"] / df["revenue"]
    df["implied_volume_mbbl"] = df["fuel_cost"] * 1e8 / (df["jet_lagged"] * df["fx"]) / 1e6
    out = ROOT / "output" / "quarterly.csv"
    df.to_csv(out, encoding="utf-8-sig")

    print("\n[검증] 분기 합계 vs 연간 공시")
    for year, total in ANNUAL_FUEL.items():
        qs = [fuel.get(f"{year}Q{q}") for q in (1, 2, 3, 4)]
        if all(v is not None for v in qs):
            print(f"  {year}: 분기합 {sum(qs):,.0f}억 vs 연간 {total:,.0f}억 → 차이 {sum(qs) - total:+,.1f}억")

    pd.set_option("display.width", 200)
    print("\n" + df.round(1).to_string())
    print(f"\n저장: {out}  ({len(df)}개 분기)")


if __name__ == "__main__":
    main()
