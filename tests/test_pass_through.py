import unittest

import numpy as np
import pandas as pd

from src.decomposition import pass_through_frame, pass_through_quarter, pass_through_regression


def synthetic(p: float, noise: float = 0.0):
    """매출이 알려진 전가율 p 로 만들어진 가상 항공사. 2019Q1~2024Q4."""
    rng = np.random.default_rng(0)
    idx = [f"{y}Q{q}" for y in range(2019, 2025) for q in range(1, 5)]
    n = len(idx)
    pax = 2 + rng.uniform(-0.5, 0.5, n)
    cargo = 100 + rng.uniform(-20, 20, n)
    jet = 80 + rng.uniform(-30, 30, n)
    fx = 1200 + rng.uniform(-100, 100, n)
    fuel = 5000 * jet * fx / (80 * 1200)
    df = pd.DataFrame({"jet_lagged": jet, "fx": fx, "fuel_cost": fuel}, index=idx)
    vol = pd.DataFrame({"pax_m": pax, "cargo_kt": cargo}, index=idx)
    # 전년 동기 대비 매출 변동이 300 + 4000·Δ여객 + 10·Δ화물 + p·가격효과 가 되도록 누적
    rev = pd.Series(30000.0, index=idx)
    for i in range(4, n):
        q, prev = idx[i], idx[i - 4]
        unit, unit_prev = jet[i] * fx[i], jet[i - 4] * fx[i - 4]
        effect = fuel[i] * (1 - unit_prev / unit)
        rev[q] = (rev[prev] + 300 + 4000 * (pax[i] - pax[i - 4]) + 10 * (cargo[i] - cargo[i - 4])
                  + p * effect + noise * rng.standard_normal())
    df["revenue"] = rev
    return df, vol


class PassThroughTest(unittest.TestCase):
    def test_recovers_known_pass_through(self):
        df, vol = synthetic(0.8)
        r = pass_through_regression(pass_through_frame(df, vol), "2020", "2024")
        self.assertEqual(r["관측치"], 20)
        self.assertAlmostEqual(r["전가율"], 0.8, places=6)
        self.assertAlmostEqual(r["여객 계수"], 4000, places=3)
        self.assertAlmostEqual(r["R2"], 1.0, places=9)

    def test_interval_contains_true_value_with_noise(self):
        df, vol = synthetic(0.9, noise=200)
        r = pass_through_regression(pass_through_frame(df, vol), "2020", "2024")
        self.assertLess(r["하한"], 0.9)
        self.assertGreater(r["상한"], 0.9)

    def test_price_effect_is_current_volume_times_unit_change(self):
        df, vol = synthetic(0.5)
        f = pass_through_frame(df, vol)
        q, p = "2021Q3", "2020Q3"
        volume = df.loc[q, "fuel_cost"] / (df.loc[q, "jet_lagged"] * df.loc[q, "fx"])
        expected = volume * (df.loc[q, "jet_lagged"] * df.loc[q, "fx"] - df.loc[p, "jet_lagged"] * df.loc[p, "fx"])
        self.assertAlmostEqual(f.loc[q, "price_effect"], expected)

    def test_single_quarter_removes_passenger_effect(self):
        df, vol = synthetic(0.7)
        f = pass_through_frame(df, vol)
        r = f.loc["2022Q2"]
        got = pass_through_quarter(f, "2022Q2", pax_coef=4000)
        self.assertAlmostEqual(got, (r["d_revenue"] - 4000 * r["d_pax"]) / r["price_effect"])


if __name__ == "__main__":
    unittest.main()
