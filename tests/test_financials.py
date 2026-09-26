import unittest

from src.financials import build_history, extract_year


def row(sj, aid, name, amount):
    return {"sj_div": sj, "account_id": aid, "account_nm": name, "thstrm_amount": amount}


# DART fnlttSinglAcntAll 응답 형식을 흉내 낸 행 (금액 단위: 원)
ROWS = [
    row("CIS", "ifrs-full_Revenue", "매출액", "16,000,000,000,000"),
    row("CIS", "dart_OperatingIncomeLoss", "영업이익", "1,500,000,000,000"),
    row("CF", "-표준계정코드 미사용-", "감가상각비", "1,800,000,000,000"),
    row("CF", "-표준계정코드 미사용-", "사용권자산감가상각비", "400,000,000,000"),
    row("CF", "-표준계정코드 미사용-", "무형자산상각비", "50,000,000,000"),
    row("CF", "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "유형자산의 취득", "-1,200,000,000,000"),
    row("CF", "ifrs-full_PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities", "무형자산의 취득", "-30,000,000,000"),
    row("BS", "ifrs-full_CashAndCashEquivalents", "현금및현금성자산", "3,000,000,000,000"),
    row("BS", "ifrs-full_CurrentAssets", "유동자산", "8,000,000,000,000"),
    row("BS", "ifrs-full_CurrentLiabilities", "유동부채", "10,000,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "단기차입금", "500,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "유동성장기부채", "1,000,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "장기차입금", "4,000,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "사채", "2,000,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "사채할인발행차금", "-10,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "유동리스부채", "700,000,000,000"),
    row("BS", "-표준계정코드 미사용-", "비유동리스부채", "3,300,000,000,000"),
    row("BS", "ifrs-full_NoncontrollingInterests", "비지배지분", "100,000,000,000"),
]


class ExtractTest(unittest.TestCase):
    def test_lines_in_eok(self):
        v = extract_year(ROWS, 2025).values
        self.assertAlmostEqual(v["revenue"], 160000)
        self.assertAlmostEqual(v["depreciation"], 22500)          # 감가 + 사용권 + 무형 합계
        self.assertAlmostEqual(v["capex"], 12300)                 # 유출(음수)을 양수로
        self.assertAlmostEqual(v["debt"], 75000)                  # 사채할인발행차금 제외
        self.assertAlmostEqual(v["current_debt"], 15000)          # 단기차입금 + 유동성장기부채
        self.assertAlmostEqual(v["lease_liabilities"], 40000)
        self.assertAlmostEqual(v["current_lease"], 7000)          # 비유동리스부채는 제외

    def test_history_and_override(self):
        hist, log = build_history({2025: ROWS}, {"fuel_cost": {"2025": 45000.0}})
        r = hist.loc[2025]
        self.assertAlmostEqual(r["ebitda"], 15000 + 22500)
        self.assertAlmostEqual(r["net_debt"], 75000 + 40000 - 30000)
        self.assertAlmostEqual(r["operating_nwc"], (80000 - 30000) - (100000 - 15000 - 7000))
        self.assertIn("override", set(log["account_id"]))

    def test_missing_required_line_raises(self):
        rows = [r for r in ROWS if r["sj_div"] != "CF"]
        with self.assertRaises(ValueError):
            build_history({2025: rows})


if __name__ == "__main__":
    unittest.main()
