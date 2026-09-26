"""유가·환율이 현금흐름에 미친 영향 분석 (본론).

output/quarterly.csv 를 읽어 요인 분해·민감도·영업이익 브리지를 계산하고,
차트와 요약 문서(output/cashflow_analysis.md)를 만든다.

실행: python analyze_cashflow.py   (먼저 python build_quarterly.py 로 데이터셋을 만들 것)
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from src.config import ROOT
from src.decomposition import add_factors, fuel_sensitivity, operating_bridge, yoy_decomposition
from src.report import BLUE, BLUE_ORDINAL, COMMA, INK, INK2, ORANGE, SURFACE, plt

FACTORS = ["물량(매출) 기여", "원유가격 기여", "환율 기여", "크랙·효율 기여"]
COLORS = dict(zip(FACTORS, ["#9ec5f4", "#2a78d6", "#1baf7a", "#eb6834"]))


def chart_decomposition(dec: pd.DataFrame, path):
    d = dec.tail(14)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(d))
    pos = np.zeros(len(d))
    neg = np.zeros(len(d))
    for f in FACTORS:
        v = d[f].values
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=bottom, width=0.62, color=COLORS[f], label=f, edgecolor=SURFACE, linewidth=1)
        pos = pos + np.where(v >= 0, v, 0)
        neg = neg + np.where(v < 0, v, 0)
    ax.plot(x, np.log1p(d["유류비 변동"].values), marker="o", markersize=7, linewidth=2,
            color=INK, label="유류비 변동(합계)")
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(x, d.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_ylabel("전년 동기 대비 (로그 기여도)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    ax.set_title("유류비 변동은 무엇으로 설명되는가 — 물량·원유가격·환율·크랙 분해")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_fuel_share(df: pd.DataFrame, path):
    f = add_factors(df)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    x = np.arange(len(f))
    a1.bar(x, f["fuel_per_revenue"], width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=1)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_ylabel("유류비 / 매출")
    a1.set_title("유류비 부담과 항공유 프리미엄(크랙) 추이")
    idx = f["crack_index"] / f["crack_index"].iloc[:4].mean()
    a2.plot(x, idx, marker="o", markersize=5, linewidth=2, color=ORANGE)
    a2.axhline(1.0, color="#c3c2b7", linewidth=1, linestyle="--")
    a2.set_ylabel("크랙 지수\n(2019년 평균 = 1)")
    a2.set_xticks(x, f.index, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_bridge(br: pd.DataFrame, path):
    d = br.tail(10)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(d))
    width = 0.26
    for i, (col, color) in enumerate(zip(["매출 증감", "유류비 증감(−)", "기타비용 증감(−)"],
                                         [BLUE_ORDINAL[1], ORANGE, "#9ec5f4"])):
        ax.bar(x + (i - 1) * width, d[col], width=width, color=color, label=col, edgecolor=SURFACE, linewidth=1)
    ax.plot(x, d["영업이익 변동"], marker="o", markersize=7, linewidth=2, color=INK, label="영업이익 변동")
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(x, d.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("전년 동기 대비 (억원)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2, loc="upper left")
    ax.set_title("영업이익은 무엇 때문에 움직였는가 — 매출·유류비·기타비용")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "output" / "quarterly.csv", index_col="quarter")
    dec, br, sens = yoy_decomposition(df), operating_bridge(df), fuel_sensitivity(df)
    charts = ROOT / "output" / "charts"
    chart_decomposition(dec, charts / "fuel_decomposition.png")
    chart_fuel_share(df, charts / "fuel_share.png")
    chart_bridge(br, charts / "operating_bridge.png")

    fmt_pct = lambda v: f"{v:+.1%}"
    dec_md = dec.drop(columns=["합계 확인"]).tail(12).map(fmt_pct)
    br_md = br.tail(8).round(0).map(lambda v: f"{v:,.0f}")
    f = add_factors(df)

    text = f"""# 유가·환율이 현금흐름에 미친 영향 — 분기 실증 분석

대한항공 연결 기준 {df.index[0]}~{df.index[-1]} ({len(df)}개 분기).
유류비는 각 정기보고서 '비용의 성격별 분류' 주석, 유가·환율은 한국은행 ECOS 분기 평균입니다.
분기 유류비 합계는 7개 연도 모두 연간 공시액과 일치합니다(검증 완료).

## 1. 유류비는 무엇으로 움직이는가

유류비를 네 요인의 곱으로 보고 전년 동기 대비 로그 기여도로 나눴습니다.

    유류비 = 물량 × 원유가격(두바이) × 환율 × 크랙·효율

![decomposition](charts/fuel_decomposition.png)

{dec_md.to_markdown()}

## 2. 민감도 (전년 동기 로그변화율 회귀, 관측치 {sens['관측치']}개)

| 설명변수 | 탄력성 | 해석 |
|---|---|---|
| 원유가격(두바이) | **{sens['원유가격 탄력성']:.2f}** | 유가가 10% 오르면 유류비는 약 {sens['원유가격 탄력성'] * 10:.1f}% 증가 |
| 환율(원/달러) | **{sens['환율 탄력성']:.2f}** | 환율이 10% 오르면 유류비는 약 {sens['환율 탄력성'] * 10:.1f}% 증가 |
| 매출(물량 대용) | **{sens['매출 탄력성']:.2f}** | 매출이 10% 늘면 유류비는 약 {sens['매출 탄력성'] * 10:.1f}% 증가 |
| 설명력 R² | {sens['R2']:.3f} | |

원유·환율 탄력성이 1보다 작습니다. 원유가격이 그대로 항공유 가격이 되는 것이 아니고(정제 마진·계약 시차),
헤지와 유류할증료가 일부를 흡수하기 때문으로 보입니다.

## 3. 영업이익 브리지 (전년 동기 대비, 억원)

![bridge](charts/operating_bridge.png)

{br_md.to_markdown()}

## 4. 항공유 프리미엄(크랙)의 존재

![fuel share](charts/fuel_share.png)

원유 환산 소비량(유류비 ÷ 원유가격 ÷ 환율)을 매출로 나눈 값은 원유 대비 항공유 가격과 연료 효율을 함께 담습니다.
2026년 2분기에는 유류비가 전년 동기 대비 {dec.loc['2026Q2', '유류비 변동']:+.0%} 늘었는데 원유가격 기여는
{dec.loc['2026Q2', '원유가격 기여']:+.0%}에 그쳤고, 나머지는 물량({dec.loc['2026Q2', '물량(매출) 기여']:+.0%}),
환율({dec.loc['2026Q2', '환율 기여']:+.0%}), 크랙·효율({dec.loc['2026Q2', '크랙·효율 기여']:+.0%})이었습니다.
**유가만 보고 항공사 원가를 추정하면 놓치는 부분이 있습니다.**

## 5. 한계

- **물량 대용치로 매출을 사용**했습니다. 매출에는 유류할증료가 들어 있어 유가와 일부 겹칩니다(순환성).
  공시되는 수송실적(ASK·RPK)은 반기·연간 단위여서 분기 분석에 바로 쓸 수 없었습니다.
- **2025년 이후 전년 동기 비교는 연결 범위가 다릅니다.** 아시아나항공이 2024년 12월 31일 인수의제일로 연결됐습니다.
- **분기 배분 오차**: 4분기 유류비는 연간에서 1~3분기를 뺀 값이고, 반기보고서의 3개월 칸을 2분기로 봤습니다.
- 크랙·효율 요인은 잔차이므로 항공유 프리미엄 외에 기재 효율, 헤지 손익 귀속, 재고 효과가 섞여 있습니다.
"""
    out = ROOT / "output" / "cashflow_analysis.md"
    out.write_text(text, encoding="utf-8")
    print(f"저장: {out}")
    print(f"차트: fuel_decomposition.png, fuel_share.png, operating_bridge.png")
    print("\n민감도:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sens.items()})


if __name__ == "__main__":
    main()
