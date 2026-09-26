"""반도체 제외 KOSPI (베타 추정용 대체 시장지수).

평가기준일 현재 삼성전자·삼성전자우·SK하이닉스가 KOSPI 시가총액의 큰 비중을 차지해,
개별 종목의 KOSPI 대비 베타가 반도체 주가 흐름에 좌우된다. 반도체를 뺀 지수를 만들어 베타를 다시 추정한다.

    R_ex(t) = (R_KOSPI(t) − w(t−1) · R_semi(t)) / (1 − w(t−1))

w 는 KOSPI 시가총액 대비 반도체 3종목 비중, R_semi 는 세 종목의 시가총액가중 수익률이다.
한계: 신규상장·유상증자 등 지수 산식상의 시가총액 조정은 반영하지 못한다.
"""
from __future__ import annotations

import pandas as pd

SEMIS = {"005930": "삼성전자", "005935": "삼성전자우", "000660": "SK하이닉스"}


def ex_semis_index(kospi_close: pd.Series, kospi_mcap: pd.Series,
                   semi_close: dict[str, pd.Series], semi_mcap: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    """(반도체 제외 지수 수준[시작일=100], 반도체 비중 w) — 일별."""
    px = pd.DataFrame(semi_close).dropna()
    caps = pd.DataFrame(semi_mcap).reindex(px.index).ffill()
    df = pd.concat([kospi_close.rename("k"), kospi_mcap.rename("kcap"), caps.sum(axis=1).rename("scap")],
                   axis=1).dropna()
    px, caps = px.reindex(df.index).ffill(), caps.reindex(df.index).ffill()
    w = (df["scap"] / df["kcap"]).rename("semi_weight")
    r_k = df["k"].pct_change()
    weights = caps.shift(1).div(caps.shift(1).sum(axis=1), axis=0)
    r_s = (px.pct_change() * weights).sum(axis=1, min_count=1)
    r_ex = ((r_k - w.shift(1) * r_s) / (1 - w.shift(1))).fillna(0.0)
    return (100 * (1 + r_ex).cumprod()).rename("KOSPI_ex_semis"), w
