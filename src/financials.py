"""DART 계정과목을 밸류에이션용 표준 항목으로 정리한다.

회사마다 계정명이 조금씩 달라서 표준계정코드(account_id)를 먼저 찾고, 없으면 계정명 패턴으로 찾는다.
어떤 계정이 어느 항목에 들어갔는지 mapping 로그로 남겨 검증할 수 있게 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

EOK = 1e8  # 원 → 억원


@dataclass(frozen=True)
class Rule:
    line: str
    statements: tuple[str, ...]
    ids: tuple[str, ...] = ()
    pattern: str | None = None
    exclude: str | None = None
    mode: str = "first"          # first: 첫 번째 일치 계정 / sum: 일치 계정 합계
    absolute: bool = False       # 현금흐름표의 유출(음수)을 양수로 바꿀 때


RULES: tuple[Rule, ...] = (
    Rule("revenue", ("IS", "CIS"), ("ifrs-full_Revenue",), r"^(매출액|영업수익|수익\(매출액\)|매출)$"),
    Rule("operating_income", ("IS", "CIS"), ("dart_OperatingIncomeLoss",), r"^영업이익(\(손실\))?$"),
    Rule("depreciation", ("CF",),
         ("ifrs-full_AdjustmentsForDepreciationExpense", "ifrs-full_AdjustmentsForAmortisationExpense",
          "ifrs-full_AdjustmentsForDepreciationAndAmortisationExpense"),
         r"(감가상각비|무형자산상각비|상각비)$", r"환입|대손", mode="sum", absolute=True),
    Rule("capex", ("CF",),
         ("ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
          "ifrs-full_PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities"),
         # 회사·연도별로 "유형자산의 취득", "유형자산 및 투자부동산의 취득" 등 표기가 달라 앞머리만 고정한다.
         # 금융자산·관계기업투자의 취득은 이름이 다른 말로 시작하므로 걸리지 않는다.
         r"^(유형자산|무형자산|투자부동산)\S*[^,]*취득$", mode="sum", absolute=True),
    Rule("cash", ("BS",), ("ifrs-full_CashAndCashEquivalents",), r"^현금및현금성자산$"),
    Rule("current_assets", ("BS",), ("ifrs-full_CurrentAssets",), r"^유동자산$"),
    Rule("current_liabilities", ("BS",), ("ifrs-full_CurrentLiabilities",), r"^유동부채$"),
    Rule("debt", ("BS",), (), r"(차입금|사채|유동성장기부채)$", r"할인|할증|이자|상환할증금", mode="sum"),
    Rule("current_debt", ("BS",), (), r"^(단기|유동)\S*(차입금|사채|장기부채)$", r"할인|할증|이자", mode="sum"),
    Rule("lease_liabilities", ("BS",), (), r"리스부채$", mode="sum"),
    Rule("current_lease", ("BS",), (), r"^(유동|유동성|단기)리스부채$", mode="sum"),
    Rule("equity_parent", ("BS",), ("ifrs-full_EquityAttributableToOwnersOfParent",), r"^지배기업\S*소유주\S*지분$"),
    Rule("nci", ("BS",), ("ifrs-full_NoncontrollingInterests",), r"^비지배지분$"),
)

REQUIRED = ("revenue", "operating_income", "depreciation", "capex", "cash", "current_assets", "current_liabilities")


def _amount(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.replace(",", "").strip()
    if text in ("", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _match(rule: Rule, rows: list[dict]) -> list[dict]:
    """표준계정코드와 계정명 패턴으로 행을 찾는다.

    mode="first" 는 표준계정코드를 우선한다.
    mode="sum" 은 둘을 합친다. 같은 항목이라도 어떤 행은 표준계정코드가 없기 때문이다.
    (예: 대한항공 2023년 현금흐름표의 '유형자산 및 투자부동산의 취득' 은 코드가 없다.)
    """
    pool = [r for r in rows if r.get("sj_div") in rule.statements]
    by_id = [r for r in pool if r.get("account_id") in rule.ids]
    if by_id and rule.mode == "first":
        return by_id
    by_name = []
    if rule.pattern is not None:
        for r in pool:
            name = re.sub(r"\s+", "", r.get("account_nm", ""))
            if re.search(rule.pattern, name) and not (rule.exclude and re.search(rule.exclude, name)):
                by_name.append(r)
    if rule.mode == "first":
        return by_name[:1]
    seen, out = set(), []
    for r in by_id + by_name:          # 같은 행이 코드·이름 양쪽에 걸리면 한 번만 센다
        if id(r) not in seen:
            seen.add(id(r))
            out.append(r)
    return out


@dataclass
class Extraction:
    values: dict[str, float] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)


def extract_year(rows: list[dict], year: int) -> Extraction:
    """사업보고서 한 개(해당 연도 당기 금액)에서 표준 항목을 뽑는다."""
    ext = Extraction()
    for rule in RULES:
        hits = _match(rule, rows)
        if rule.mode == "first":
            hits = hits[:1]
        total, used = 0.0, 0
        for r in hits:
            amt = _amount(r.get("thstrm_amount"))
            if amt is None:
                continue
            amt = abs(amt) if rule.absolute else amt
            total += amt
            used += 1
            ext.log.append({"year": year, "line": rule.line, "sj_div": r.get("sj_div"),
                            "account_id": r.get("account_id"), "account_nm": r.get("account_nm"),
                            "amount_eok": round(amt / EOK, 1)})
        if used:
            ext.values[rule.line] = total / EOK
    return ext


def build_history(reports: dict[int, list[dict]], overrides: dict[str, dict] | None = None,
                  proforma: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """연도별 DART 행 → (표준 재무 DataFrame[억원], 매핑 로그 DataFrame).

    overrides: {"fuel_cost": {"2024": 12345.0}, "depreciation": {...}} — 주석에서 직접 입력한 값(억원).
    proforma: {"2024": {"revenue": 83186.0, ...}} — 피취득회사 실적을 손익 항목에 더한다(억원).
              재무상태표 항목은 이미 연결돼 있으므로 더하지 않는다.
    """
    overrides = overrides or {}
    records, logs = {}, []
    for year, rows in sorted(reports.items()):
        ext = extract_year(rows, year)
        records[year] = ext.values
        logs.extend(ext.log)
    hist = pd.DataFrame.from_dict(records, orient="index").sort_index()
    hist.index.name = "year"

    for line, by_year in overrides.items():
        for y, v in by_year.items():
            if int(y) in hist.index:
                hist.loc[int(y), line] = float(v)
                logs.append({"year": int(y), "line": line, "sj_div": "수기입력", "account_id": "override",
                             "account_nm": "config.toml overrides", "amount_eok": float(v)})

    for y, lines in (proforma or {}).items():
        year = int(y)
        if year not in hist.index:
            continue
        for line, add in lines.items():
            before = float(hist.loc[year, line]) if line in hist.columns and pd.notna(hist.loc[year, line]) else 0.0
            hist.loc[year, line] = before + float(add)
            logs.append({"year": year, "line": line, "sj_div": "pro forma", "account_id": "proforma",
                         "account_nm": f"피취득회사 합산 (공시 {before:,.0f} + {float(add):,.0f})", "amount_eok": float(add)})

    missing = [c for c in REQUIRED if c not in hist.columns or hist[c].isna().any()]
    if missing:
        raise ValueError(
            f"DART 재무제표에서 필수 항목을 찾지 못했습니다: {missing}. "
            "감가상각비처럼 주석에만 있는 항목은 config.toml 의 [overrides] 에 직접 입력하세요."
        )
    for col in ("debt", "current_debt", "lease_liabilities", "current_lease", "nci"):
        if col not in hist.columns:
            hist[col] = 0.0
    hist = hist.fillna({"debt": 0.0, "current_debt": 0.0, "lease_liabilities": 0.0, "current_lease": 0.0, "nci": 0.0})
    return add_derived(hist), pd.DataFrame(logs)


def add_derived(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    h["ebitda"] = h["operating_income"] + h["depreciation"]
    h["operating_nwc"] = (h["current_assets"] - h["cash"]) - (h["current_liabilities"] - h["current_debt"] - h["current_lease"])
    h["net_debt"] = h["debt"] + h["lease_liabilities"] - h["cash"]
    if "fuel_cost" in h.columns:
        h["nonfuel_cash_opex"] = h["revenue"] - h["ebitda"] - h["fuel_cost"]
    return h
