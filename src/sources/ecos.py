"""한국은행 경제통계시스템(ECOS) Open API.

사용 통계표 (2026-09 기준 코드 확인):
  817Y002 시장금리(일별)          010210000 국고채(10년), 010300000 회사채(3년, AA-)
  731Y001 주요국 통화의 대원화환율  0000001 원/미국달러(매매기준율)
  902Y003 국제상품가격(월)        010101 WTI, 010102 Dubai, 010103 Brent
  802Y001 주식시장(일별)          0001000 KOSPI지수, 0183000 시가총액(유가증권시장)
"""
from __future__ import annotations

import pandas as pd

from . import http

from .cache import cached_json

BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
PAGE = 1000

SERIES = {
    "ktb10y": ("817Y002", "D", "010210000"),
    "corp_aa3y": ("817Y002", "D", "010300000"),
    "usdkrw": ("731Y001", "D", "0000001"),
    "kospi": ("802Y001", "D", "0001000"),
    "kospi_mcap": ("802Y001", "D", "0183000"),   # 유가증권시장 시가총액(억원)
}
OIL_TABLE = "902Y003"


class EcosError(RuntimeError):
    pass


def _fetch(key: str, stat: str, cycle: str, start: str, end: str, item: str) -> list[dict]:
    rows: list[dict] = []
    begin = 1
    while True:
        url = f"{BASE}/{key}/json/kr/{begin}/{begin + PAGE - 1}/{stat}/{cycle}/{start}/{end}/{item}"
        body = http.get(url, secrets=[key]).json()
        if "StatisticSearch" not in body:
            result = body.get("RESULT", {})
            if result.get("CODE") == "INFO-200" and rows:  # 마지막 페이지 이후
                break
            raise EcosError(f"ECOS {stat}/{item}: {result.get('CODE')} {result.get('MESSAGE')}")
        block = body["StatisticSearch"]
        rows.extend(block["row"])
        if len(rows) >= int(block["list_total_count"]):
            break
        begin += PAGE
    return rows


def _to_series(rows: list[dict], cycle: str, name: str) -> pd.Series:
    fmt = {"D": "%Y%m%d", "M": "%Y%m"}[cycle]
    s = pd.Series(
        [float(r["DATA_VALUE"]) for r in rows],
        index=pd.to_datetime([r["TIME"] for r in rows], format=fmt),
        name=name,
    )
    return s.sort_index()


def get_series(key: str, name: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    """name: SERIES 의 키. start/end: YYYYMMDD."""
    stat, cycle, item = SERIES[name]
    payload = cached_json(
        "ecos", {"stat": stat, "item": item, "cycle": cycle, "start": start, "end": end},
        lambda: _fetch(key, stat, cycle, start, end, item), refresh,
    )
    return _to_series(payload["data"], cycle, name)


def get_oil_monthly(key: str, item: str, start: str, end: str, refresh: bool = False) -> pd.Series:
    """국제유가 월평균 (달러/배럴). start/end: YYYYMM."""
    payload = cached_json(
        "ecos", {"stat": OIL_TABLE, "item": item, "cycle": "M", "start": start, "end": end},
        lambda: _fetch(key, OIL_TABLE, "M", start, end, item), refresh,
    )
    return _to_series(payload["data"], "M", "oil")
