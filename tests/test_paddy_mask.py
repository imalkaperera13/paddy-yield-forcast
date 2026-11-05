"""Step 2 tests that need no Earth Engine: season windows and the DCS agreement metrics."""

import datetime as dt

import pandas as pd
import pytest

from src.config import load_config
from src.data.paddy_mask import season_windows
from scripts.mask_validation import agreement


def test_windows_follow_season_anchor():
    cfg = load_config()
    w = season_windows(cfg, "Maha", 2020)
    assert w["flood"][0] == dt.date(2020, 9, 15)
    assert w["harvest"][1] == dt.date(2021, 4, 13)
    for s in ("Maha", "Yala"):
        w = season_windows(cfg, s, 2010)
        assert w["flood"][0] < w["grow"][0] < w["harvest"][0]
        assert all(a < b for a, b in w.values())


def test_agreement_perfect_match():
    s = pd.Series([100.0, 200.0, 300.0])
    r = agreement(s, s)
    assert r["n"] == 3
    assert r["bias_pct"] == pytest.approx(0)
    assert r["mape_pct"] == pytest.approx(0)
    assert r["pearson_r"] == pytest.approx(1)


def test_agreement_ignores_missing_and_zero_dcs():
    m = pd.Series([50.0, 80.0, None, 10.0])
    d = pd.Series([100.0, 100.0, 100.0, 0.0])
    r = agreement(m, d)
    assert r["n"] == 2
    assert r["ratio_median"] == pytest.approx(0.65)
    assert r["bias_pct"] == pytest.approx(-35)
