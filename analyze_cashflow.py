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
from src.decomposition import (add_factors, fuel_sensitivity, operating_bridge, pass_through_frame,
                               pass_through_quarter, pass_through_regression, yoy_decomposition)
from src.report import BLUE, BLUE_ORDINAL, COMMA, INK, INK2, ORANGE, SURFACE, plt

FACTORS = ["물량(매출) 기여", "항공유가격 기여", "환율 기여", "잔차 기여"]
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
    ax.set_title("유류비 변동은 무엇으로 설명되는가 — 물량·항공유가격·환율·잔차 분해")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_crack(df: pd.DataFrame, path):
    """항공유와 원유 가격, 그리고 그 배수(크랙)."""
    f = add_factors(df)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    x = np.arange(len(f))
    a1.plot(x, f["jet"], linewidth=2, color=ORANGE, marker="o", markersize=4, label="항공유 현물 (미 걸프연안)")
    a1.plot(x, f["oil"], linewidth=2, color=BLUE, marker="o", markersize=4, label="두바이유")
    a1.set_ylabel("달러/배럴")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2)
    a1.set_title("항공유는 원유와 따로 움직인다 — 크랙(정제마진)")
    a2.bar(x, f["crack_ratio"], width=0.62, color="#86b6ef", edgecolor=SURFACE, linewidth=1)
    a2.axhline(f["crack_ratio"].mean(), color=INK2, linewidth=1, linestyle="--")
    a2.set_ylabel("항공유 ÷ 원유 (배)")
    a2.set_ylim(0.9, max(1.7, f["crack_ratio"].max() * 1.05))
    a2.set_xticks(x, f.index, rotation=45, ha="right")
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
    a1.set_title("유류비 부담과 환산 소비량")
    a2.plot(x, f["jet_equiv_bbl"] / 1e6, marker="o", markersize=5, linewidth=2, color=ORANGE)
    a2.set_ylabel("환산 소비량\n(백만 배럴/분기)")
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
    chart_crack(df, charts / "crack.png")
    chart_bridge(br, charts / "operating_bridge.png")

    dec_md = dec.drop(columns=["합계 확인"]).tail(12).copy()
    for col in dec_md.columns:
        dec_md[col] = (dec_md[col].map(lambda v: f"{v:.2f}배") if col == "크랙 배수"
                       else dec_md[col].map(lambda v: f"{v:+.1%}"))
    br_md = br.tail(8).round(0).map(lambda v: f"{v:,.0f}")
    f = add_factors(df)

    last = df.loc["2025Q1":"2025Q4"]
    vol_y = f.loc["2025Q1":"2025Q4", "jet_equiv_bbl"].sum()
    fx_y, jet_y = last["fx"].mean(), last["jet_lagged"].mean()
    fuel_y = last["fuel_cost"].sum()
    q2 = dec.loc["2026Q2"]

    # 유류할증료 전가율: 연결 범위와 화물 정의가 같은 2019~2024년 데이터(전년 대비 2020~2024년)로 추정
    volume = pd.read_csv(ROOT / "reference" / "icn_volume_quarterly.csv", index_col="quarter", encoding="utf-8-sig")
    pt_frame = pass_through_frame(df, volume)
    samples = [("2019~2024 전체", "2020"), ("2020년 제외", "2021"), ("2022년 이후", "2022")]
    pt = {label: pass_through_regression(pt_frame, start, "2024") for label, start in samples}
    pt_full = pt["2019~2024 전체"]
    pt_q2 = pass_through_quarter(pt_frame, "2026Q2", pt_full["여객 계수"])
    pt_md = "\n".join(f"| {label} | {r['관측치']} | {r['전가율']:.2f} | {r['하한']:.2f}~{r['상한']:.2f} | "
                      f"{r['t']:.1f} | {r['R2']:.2f} |" for label, r in pt.items())

    text = f"""# 유가·환율이 현금흐름에 미친 영향 — 분기 실증 분석

대한항공 연결 기준 {df.index[0]}~{df.index[-1]} ({len(df)}개 분기).
유류비는 각 정기보고서 '비용의 성격별 분류' 주석에서 뽑았고, 분기 합계가 7개 연도 모두 연간 공시액과 일치합니다.
항공유 가격은 미국 걸프연안 현물(FRED DJFUELUSGULF)에 4주 시차를 준 분기 평균, 원유·환율은 한국은행 ECOS입니다.

## 1. 유류비는 무엇으로 움직이는가

항등식으로 네 요인에 나눴습니다. 로그 변화율이라 기여도가 그대로 더해집니다.

    유류비 = 물량 × 항공유가격 × 환율 × 잔차
    (잔차 = 연료 효율, 매출-물량 괴리, 매입 시차, 헤지 손익 귀속)

![decomposition](charts/fuel_decomposition.png)

{dec_md.to_markdown()}

가장 최근인 2026년 2분기를 보면, 유류비가 전년 동기 대비 {q2['유류비 변동']:+.0%} 늘었고 그중
**항공유 가격 기여가 {q2['항공유가격 기여']:+.0%}p**로 대부분입니다. 물량 {q2['물량(매출) 기여']:+.0%}p,
환율 {q2['환율 기여']:+.0%}p, 잔차 {q2['잔차 기여']:+.0%}p였습니다.

## 2. 민감도 — 항등식으로 직접 계산

유류비는 항등식이므로 탄력성을 추정할 필요가 없습니다. 2025년 연결 실적 기준으로 환산하면 이렇습니다.

| 변화 | 연간 유류비 영향 | 2025년 영업이익({last['operating_income'].sum():,.0f}억원) 대비 |
|---|---|---|
| 항공유 1달러/배럴 상승 | {vol_y * fx_y / 1e8:,.0f}억원 | {vol_y * fx_y / 1e8 / last['operating_income'].sum():.0%} |
| 항공유 10% 상승 | {fuel_y * 0.1:,.0f}억원 | {fuel_y * 0.1 / last['operating_income'].sum():.0%} |
| 환율 10원 상승 | {vol_y * jet_y * 10 / 1e8:,.0f}억원 | {vol_y * jet_y * 10 / 1e8 / last['operating_income'].sum():.0%} |

환산 소비량은 2025년 연결 기준 **{vol_y / 1e6:,.1f}백만 배럴**입니다.
회사는 사업보고서에 "연간 예상 소모량 약 3,050만 배럴, 유가 1달러 변동 시 약 3,050만 달러"라고 공시하는데,
이는 대한항공 단독 기준입니다. 아시아나를 포함한 연결 기준으로 환산하면 {vol_y / 1e6 / 30.5:.1f}배가 되고,
같은 방식으로 계산한 민감도가 위 표입니다. **공시된 민감도를 외부 데이터로 재현하고 연결 기준으로 확장한 것입니다.**

환율의 경우 유류비만 보면 10원당 {vol_y * jet_y * 10 / 1e8:,.0f}억원이지만, 회사 공시 순노출은
"연간 달러 부족량 약 16억 달러, 환율 10원 변동 시 약 160억원"입니다. 달러 매출이 상당 부분을 상쇄하기 때문입니다.

## 3. 영업이익은 무엇 때문에 움직였나

![bridge](charts/operating_bridge.png)

{br_md.to_markdown()}

2026년 2분기 영업적자({df.loc['2026Q2', 'operating_income']:,.0f}억원)는 유류비가 직접 원인입니다.
매출이 {br.loc['2026Q2', '매출 증감']:,.0f}억원 늘었지만 유류비가 {-br.loc['2026Q2', '유류비 증감(−)']:,.0f}억원 늘어
상쇄되지 못했습니다.

## 4. 항공유는 원유와 따로 움직인다

![crack](charts/crack.png)

> **크랙 배수** = 항공유 현물가격(미 걸프연안) ÷ 두바이유 가격, 둘 다 달러/배럴 분기 평균.
> 정제마진(크랙 스프레드)을 차이 대신 비율로 나타낸 것으로, 1.60배는 항공유가 원유보다 60% 비싸다는 뜻입니다.
> 분자와 분모의 지역이 달라 수준보다는 확대·축소 방향을 보는 지표입니다.

원유 대비 항공유 가격(크랙 배수)은 2020년 2분기 1.0배에서 2026년 2분기 **{f.loc['2026Q2', 'crack_ratio']:.2f}배**까지 벌어졌습니다.
2023년 1분기에는 원유가 전년 대비 내렸는데도 크랙이 1.57배로 확대돼 유류비 부담이 유지됐습니다.

**원유 선물로 헤지해도 이 부분은 남습니다(베이시스 리스크).** 회사는 연간 소모량의 최대 50%를
Zero Cost Collar 등으로 헤지한다고 공시하는데, 기초자산이 원유라면 크랙 확대는 막지 못합니다.

![fuel share](charts/fuel_share.png)

항공유 기준 환산 소비량은 분기 8~13백만 배럴 범위에서 안정적입니다. 공시된 단독 소모량(분기 약 7.6백만 배럴)에
자회사와 아시아나를 더한 규모와 맞아떨어져, 가격 대용치 선택이 타당했음을 뒷받침합니다.

## 5. 유류할증료는 얼마나 전가되나

유가가 오르면 항공사는 유류할증료로 일부를 운임에 넘깁니다. 전년 동기 대비 변화로 전가율 p를 추정했습니다.

    Δ매출 = c + a·Δ여객 + b·Δ화물 + p·(유류비 가격효과)
    유류비 가격효과 = 환산 소비량 × (항공유 4주 시차 × 환율) 변동 = 유류비 × (1 − 전년 단가 / 당해 단가)

가격효과는 소비량을 당분기로 고정해 물량 변화를 뺀 순수 단가 요인이고, 물량은 인천공항 국제선 출발 운송실적
(연결 대상 항공사)의 여객·화물로 통제했습니다. 연결 범위와 화물 정의가 같은 2019~2024년 자료만 써서
관측치는 2020~2024년 분기입니다.

| 표본 | 관측치 | 전가율 p | 95% 구간 | t | R² |
|---|---|---|---|---|---|
{pt_md}
| 2026Q2 단일 분기 | 1 | {pt_q2:.2f} | — | — | — |

2026Q2 단일 분기는 매출 증가 {pt_frame.loc['2026Q2', 'd_revenue']:,.0f}억원에서 여객 효과(전체 표본 계수
{pt_full['여객 계수']:,.0f}억원/백만 명 × Δ여객 {pt_frame.loc['2026Q2', 'd_pax']:+.2f}백만 명)를 빼고
가격효과 {pt_frame.loc['2026Q2', 'price_effect']:,.0f}억원으로 나눈 값입니다. 2025년부터 화물 정의가 달라 화물은 빼고,
상수(추세)도 빼서 여객으로 설명되지 않는 매출 변동을 모두 전가로 봤습니다.

**평상시에는 거의 전액 전가됩니다.** 다만 2021~2022년은 유가 급등과 코로나 이후 운임 급등이 겹쳐 추정치가
위로 치우쳤을 수 있고, 급등 국면인 2026년 2분기에는 {pt_q2:.2f}으로 낮습니다. 할증료는 발권 시점 기준으로
월 단위 조정되므로 급등기에는 전가가 늦어집니다. 2022년 이후 표본은 관측치가 12개뿐이고 여객 계수가
{pt['2022년 이후']['여객 계수']:,.0f}로 전체 표본({pt_full['여객 계수']:,.0f})보다 크게 작아, 물량 통제가 약한 추정입니다.
모델과 대시보드는 코로나 영향이 적은 2022년 이후 추정치({pt['2022년 이후']['전가율']:.2f})를 반올림한 0.9를 기본값으로 씁니다.

## 6. 부록 — 회귀 계수를 쓰지 않은 이유

같은 데이터로 `Δln(유류비) ~ Δln(항공유) + Δln(환율) + Δln(매출)` 회귀를 돌리면
항공유 {sens['jet_lagged 가격 계수']:.2f}, 환율 {sens['환율 계수']:.2f}, 매출 {sens['매출 계수']:.2f}가 나옵니다(R² {sens['R2']:.3f}).
**항등식상 계수는 모두 1이어야 하므로, 1에서 벗어난 만큼은 잔차 요인이 각 변수와 함께 움직인 정도일 뿐입니다.**
표본을 바꾸면 환율 계수가 음수까지 내려갈 정도로 불안정해서, 구조적 탄력성으로 제시하지 않았습니다.
민감도는 2장처럼 항등식으로 계산하는 편이 정확합니다.

## 7. 한계

- **물량 대용치로 매출을 사용**했습니다. 매출에는 유류할증료가 들어 있어 가격 요인과 일부 겹칩니다.
  공시 수송실적(ASK·RPK)은 반기·연간 단위여서 분기 분석에 바로 쓸 수 없었습니다.
- **항공유 가격은 미국 걸프연안 현물**입니다. 실제 매입 기준인 싱가포르 MOPS는 유료 데이터입니다.
  4주 시차를 주면 잔차의 평균 절대값이 17.9%에서 14.5%로 줄어 이 값을 채택했습니다.
- **2025년 이후 전년 동기 비교는 연결 범위가 다릅니다.** 아시아나항공이 2024년 12월 31일 인수의제일로 연결됐습니다.
- **분기 배분 오차**: 4분기 유류비는 연간에서 1~3분기를 뺀 값이고, 반기보고서의 3개월 칸을 2분기로 봤습니다.
- 잔차에는 헤지 손익 귀속과 재고 효과가 섞여 있어, 순수한 연료 효율로 읽으면 안 됩니다.
- **전가율의 물량 자료는 인천공항 국제선만** 포함합니다. 화물 항목이 2024년까지 `항공화물`, 2025년부터 `직화물`로
  정의가 달라 회귀는 2019~2024년으로 한정했습니다.
"""
    out = ROOT / "output" / "cashflow_analysis.md"
    out.write_text(text, encoding="utf-8")
    print(f"저장: {out}")
    print("전가율:", {k: round(float(r["전가율"]), 3) for k, r in pt.items()}, "2026Q2:", round(float(pt_q2), 3))
    print("차트: fuel_decomposition.png, crack.png, fuel_share.png, operating_bridge.png")
    print("\n민감도:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sens.items()})


if __name__ == "__main__":
    main()
