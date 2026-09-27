# 외부 참조 데이터

## jet_fuel_spot_usgulf.csv

- 내용: 항공유(Kerosene-Type Jet Fuel) 현물가격, 미국 걸프연안 FOB, 달러/갤런, 일별 1990~
- 출처: FRED (세인트루이스 연준) 시리즈 `DJFUELUSGULF` — 원자료는 미국 에너지정보청(EIA)
- 갱신: `curl -o reference/jet_fuel_spot_usgulf.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF"`
- 사용 이유: 항공사 원가는 원유가 아니라 항공유 가격에 좌우된다. 아시아 기준가격인
  싱가포르 MOPS 는 유료 데이터(플라츠)여서, 공개된 미국 걸프연안 현물가격을 대용치로 쓴다.
- 한계: 지역이 다르므로 수준(level)에 차이가 있을 수 있다. 다만 변동 방향과 크랙 확대·축소
  국면은 같이 움직인다. 국내 유류할증료 단계로 역산한 MOPS 수준과 비교하면 2026년 4월 기준
  단계 18(MOPS 320~330센트/갤런 ≈ 134~139달러/배럴)과 같은 분기 이 시리즈의 평균
  152.6달러/배럴로, 방향은 일치하고 수준은 차이가 있다.
