"""공공데이터포털 금융위원회_주식시세정보 (일별 종가, 상장주식수, 시가총액)."""
from __future__ import annotations

import pandas as pd

from . import http
from .cache import cached_json

# 2025년 이후 엔드포인트는 경로에 /service/ 가 없는 _V2 형식이다.
# 옛 주소(.../service/GetStockSecuritiesInfoService/getStockPriceInfo)는 SERVICE_KEY_IS_NOT_REGISTERED 를 돌려준다.
STOCK_URL = "https://apis.data.go.kr/1160100/GetStockSecuritiesInfoService_V2/getStockPriceInfo_V2"
PAGE = 1000


class DataGoKrError(RuntimeError):
    pass


def _fetch(key: str, stock_code: str, start: str, end: str) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        params = {
            "serviceKey": key, "resultType": "json", "numOfRows": PAGE, "pageNo": page,
            "likeSrtnCd": stock_code, "beginBasDt": start, "endBasDt": end,
        }
        resp = http.get(STOCK_URL, params)
        try:
            body = resp.json()["response"]
        except (ValueError, KeyError):
            raise DataGoKrError(f"주식시세 API 응답 오류: {resp.text[:200]}") from None
        if body["header"]["resultCode"] != "00":
            raise DataGoKrError(f"주식시세 API: {body['header']['resultMsg']}")
        batch = body["body"]["items"]["item"] if body["body"]["items"] else []
        items.extend(batch)
        if len(items) >= int(body["body"]["totalCount"]) or not batch:
            break
        page += 1
    # likeSrtnCd 는 부분일치이므로 정확히 같은 종목만 남긴다
    return [i for i in items if i["srtnCd"] == stock_code]


def stock_prices(key: str, stock_code: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """index=날짜, columns=close, shares, market_cap(억원).

    endBasDt 는 종료일을 포함하지 않으므로 하루를 더해 호출한다.
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    payload = cached_json(
        "datagokr", {"code": stock_code, "start": start, "end": end_exclusive},
        lambda: _fetch(key, stock_code, start, end_exclusive), refresh,
    )
    rows = payload["data"]
    if not rows:
        raise DataGoKrError(f"{stock_code} 주가 데이터가 없습니다.")
    df = pd.DataFrame({
        "close": [float(r["clpr"]) for r in rows],
        "shares": [float(r["lstgStCnt"]) for r in rows],
        "market_cap": [float(r["mrktTotAmt"]) / 1e8 for r in rows],
    }, index=pd.to_datetime([r["basDt"] for r in rows], format="%Y%m%d"))
    return df.sort_index()
