"""Tests for the conformal, earliness and ablation code (SYNTHETIC data only)."""

import numpy as np
import pandas as pd

from src.config import load_config
from src.experiments.ablation import variants
from src.experiments.conformal import conformal_q, run_conformal
from tests.test_models import synthetic_master

CFG = load_config()


def test_conformal_quantile_finite_sample_rule():
    r = np.arange(1, 10, dtype=float)               # n = 9
    assert conformal_q(r, 0.1) == 9.0               # ceil(10 * 0.9) = 9th smallest
    assert conformal_q(r, 0.5) == 5.0               # ceil(10 * 0.5) = 5th smallest
    assert conformal_q(np.arange(1, 5, dtype=float), 0.1) == float("inf")  # k = 5 > n = 4
    assert conformal_q(np.array([np.nan, 2.0, 1.0]), 0.5) == 2.0


def test_SYNTHETIC_conformal_covers_close_to_target(tmp_path):
    df = synthetic_master(seed=3)
    cfg = {**CFG, "conformal": {**CFG["conformal"], "reliability_alphas": [0.1]}}
    p = run_conformal(df, ["f1", "f2"], "ridge", 42, cfg, tmp_path, log=lambda *_: None)
    q = p["q_standard_0.1"]
    cov = ((p.y_true >= p.y_pred - q) & (p.y_true <= p.y_pred + q)).mean()
    assert 0.8 <= cov <= 1.0, cov


def test_SYNTHETIC_conformal_never_uses_test_season_for_calibration(tmp_path):
    """Poisoning the test seasons' targets must not change any fold's quantile."""
    df = synthetic_master(seed=4)
    cfg = {**CFG, "conformal": {**CFG["conformal"], "reliability_alphas": [0.1]}}
    p0 = run_conformal(df, ["f1", "f2"], "baseline_last5", 1, cfg, tmp_path / "a",
                       log=lambda *_: None)
    last = p0.season_order.max()
    df2 = df.copy()
    df2.loc[df2.season_order == last, "yield_kg_ha"] = 1e7
    p1 = run_conformal(df2, ["f1", "f2"], "baseline_last5", 1, cfg, tmp_path / "b",
                       log=lambda *_: None)
    a = p0[p0.season_order == last]["q_standard_0.1"].to_numpy()
    b = p1[p1.season_order == last]["q_standard_0.1"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_ablation_variants_partition_groups():
    groups = {"lagged_yield": ["y_a", "y_b"], "rainfall": ["rain_1"], "modis": ["ndvi_1"],
              "temp_radiation": ["tmax_1"]}
    v = variants(groups)
    assert set(v["all"]) == {"y_a", "y_b", "rain_1", "ndvi_1", "tmax_1"}
    assert v["lagged_yield_only"] == ["y_a", "y_b"]
    assert "rain_1" not in v["all_minus_rainfall"] and "y_a" in v["all_minus_rainfall"]
    assert not any(c.startswith("y_") for c in v["no_lagged_yield"])
    assert len(v) == 9


def test_shock_list_has_named_and_data_chosen_seasons():
    from src.analysis.shocks import shock_list
    labels = pd.DataFrame({
        "district": ["A"] * 30, "season": ["Maha"] * 30, "season_year": range(2000, 2030),
        "yield_kg_ha": [4000.0] * 30, "harvested_ha": [1.0] * 30})      # SYNTHETIC
    labels.loc[labels.season_year == 2015, "yield_kg_ha"] = 3000.0
    s = shock_list(CFG, labels)
    assert {"policy/economic (named)", "weather (named)"} <= set(s.shock_type)
    assert ((s.season == "Maha") & (s.year == 2015)).any()
