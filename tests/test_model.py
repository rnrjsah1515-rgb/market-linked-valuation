import unittest

import numpy as np
import pandas as pd

from src.model import Assumptions, dcf, fx_adjusted_net_debt, project
from src.risk import hedge_analysis, simulate_paths


def flat_assumptions(**kw) -> Assumptions:
    """성장 0, 운전자본 0, CAPEX = 감가상각비인 단순한 회사. 연말 할인, 평가일 = 기준연도 말."""
    base = dict(
        company="test", base_year=2025, valuation_date=pd.Timestamp("2025-12-31"), years=[2026, 2027, 2028],
        growth=np.zeros(3), terminal_growth=0.0, mid_year=False, tax_rate=0.25,
        revenue_last=1000.0, fuel_volume_last=1e6, fx_ref=1300.0, oil_ref=80.0, fuel_pass_through=0.0,
        usd_revenue_share=0.0, usd_nonfuel_cost_share=0.0, nonfuel_ratio=0.5, da_ratio=0.1, capex_ratio=0.1,
        capex_floor_to_da=1.0, nwc_ratio=0.0, nwc_last=0.0, terminal_capex_to_da=1.0,
        net_debt=500.0, nci=0.0, shares=1e7, oil_curve=np.full(3, 80.0), fx_curve=np.full(3, 1300.0),
        curve_source="test",
    )
    base.update(kw)
    return Assumptions(**base)


class DcfTest(unittest.TestCase):
    def test_flat_company_equals_perpetuity(self):
        a = flat_assumptions(revenue_last=5000.0)
        p = project(a)
        fuel = 1e6 * 80 * 1300 / 1e8                       # 소비량 × 유가 × 환율 = 1,040억원
        self.assertAlmostEqual(p["fuel_cost"][0, 0], fuel)
        ebit = 5000 - 2500 - 500 - fuel                    # 매출 - 비유류비 - 감가상각비 - 유류비
        fcff = ebit * 0.75                                 # CAPEX = 감가상각비, 운전자본 변동 없음
        self.assertAlmostEqual(p["fcff"][0, 0], fcff)
        # 성장 0 인 영구현금흐름 → EV = FCFF / WACC (Actual/365.25 일수 계산 때문에 0.01% 이내 차이 허용)
        res = dcf(a, 0.08)
        self.assertAlmostEqual(res["ev"][0], fcff / 0.08, delta=fcff / 0.08 * 1e-4)
        self.assertAlmostEqual(res["equity"][0], res["ev"][0] - 500)

    def test_pass_through_offsets_oil_cost(self):
        a = flat_assumptions(revenue_last=5000.0, fuel_pass_through=1.0)
        up = project(a, oil=np.full(3, 100.0))["ebitda"][0, 0]
        flat = project(a)["ebitda"][0, 0]
        self.assertAlmostEqual(up, flat)                   # 100% 전가면 유가가 올라도 EBITDA 불변

    def test_stub_period_counts_remaining_fraction(self):
        a = flat_assumptions(revenue_last=5000.0, valuation_date=pd.Timestamp("2026-07-02"))
        res = dcf(a, 0.08)
        self.assertAlmostEqual(res["frac"][0], (pd.Timestamp("2026-12-31") - pd.Timestamp("2026-07-02")).days / 365.25)
        self.assertEqual(res["frac"][1], 1.0)

    def test_usd_net_debt_revalued_to_spot(self):
        # 재무상태표는 1,500원에 환산됐고 평가일 현물은 1,300원 → 달러부채 10억불이면 순차입금 2,000억원 감소
        a = flat_assumptions(revenue_last=5000.0, usd_net_debt=1e9,
                             fx_at_balance_sheet=1500.0, fx_spot=1300.0)
        base = dcf(a, 0.08)
        self.assertAlmostEqual(float(base["net_debt"][0]), 500.0 + 1e9 * (1300 - 1500) / 1e8)
        # 환율이 10% 오르면 현물도 10% 올라 순차입금이 늘고 주주가치가 줄어든다
        up = dcf(a, 0.08, fx=a.fx_curve * 1.1)
        self.assertAlmostEqual(float(up["net_debt"][0]), 500.0 + 1e9 * (1300 * 1.1 - 1500) / 1e8)
        self.assertLess(float(up["equity"][0]) - float(up["ev"][0]), float(base["equity"][0]) - float(base["ev"][0]))

    def test_no_usd_net_debt_keeps_book_value(self):
        a = flat_assumptions(revenue_last=5000.0)
        self.assertAlmostEqual(float(fx_adjusted_net_debt(a, np.array([1500.0]))[0]), 500.0)

    def test_wacc_must_exceed_growth(self):
        with self.assertRaises(ValueError):
            dcf(flat_assumptions(), 0.02, g=0.03)


class SimulationTest(unittest.TestCase):
    def test_paths_are_unbiased_around_curve(self):
        a = flat_assumptions()
        oil, fx = simulate_paths(a, 0.3, 0.08, -0.2, 200_000, 1)
        np.testing.assert_allclose(oil.mean(axis=0), a.oil_curve, rtol=0.01)
        np.testing.assert_allclose(fx.mean(axis=0), a.fx_curve, rtol=0.005)

    def test_hedge_reduces_ebitda_volatility(self):
        a = flat_assumptions(revenue_last=5000.0)
        oil, fx = simulate_paths(a, 0.3, 0.0, 0.0, 5000, 2)
        table, _, _, _ = hedge_analysis(a, 0.08, oil, fx, [0.0, 0.5, 1.0], hedge_years=2)
        std = table.filter(like="표준편차").iloc[:, 0].values
        self.assertTrue(std[0] > std[1] > std[2])
        self.assertAlmostEqual(std[2], 0.0, places=6)       # 100% 헤지 + 환율 고정이면 변동 없음


if __name__ == "__main__":
    unittest.main()
