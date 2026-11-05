"""Leakage tests for the static paddy mask (no Earth Engine needed)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.data import static_mask as sm


@pytest.fixture
def cfg():
    return load_config()


def first_test_order(cfg):
    ft = cfg["features"]["paddy_mask"]["static"]["first_test"]
    return sm.season_order(ft["season"], ft["season_year"])


def test_season_order_matches_data_convention():
    d = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    for r in d.sample(50, random_state=0).itertuples():
        assert sm.season_order(r.season, r.season_year) == r.season_order


@pytest.mark.parametrize("season", ["Maha", "Yala"])
def test_source_seasons_strictly_before_first_test(cfg, season):
    src = sm.static_source_seasons(cfg, season)
    assert src, "no source seasons"
    assert all(s == season for s, _ in src)
    assert max(sm.season_order(s, y) for s, y in src) < first_test_order(cfg)


def test_first_test_season_matches_pilot_fold0(cfg):
    p = REPO_ROOT / "results" / "tables" / "per_fold_metrics.csv"
    if not p.exists():
        pytest.skip("pilot fold table not present")
    f = pd.read_csv(p)
    f0 = f.sort_values("fold").iloc[0]
    ft = cfg["features"]["paddy_mask"]["static"]["first_test"]
    assert (f0.season, int(f0.season_year)) == (ft["season"], ft["season_year"])


def test_source_years_past_first_test_are_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["features"]["paddy_mask"]["static"]["years"] = [2000, 2009]
    with pytest.raises(ValueError):
        sm.static_source_seasons(bad, "Maha")


def test_detection_count_only_calls_source_seasons(cfg, monkeypatch):
    """The Earth Engine graph is built only from the source seasons."""
    calls = []

    class Img:
        def toInt(self):
            return self

    class Coll:
        def __init__(self, imgs):
            self.imgs = imgs

        def sum(self):
            return self

        def rename(self, _):
            return self

    class FakeEE:
        @staticmethod
        def ImageCollection(imgs):
            return Coll(imgs)

    def fake_build_mask(_ee, _cfg, season, year):
        calls.append((season, year))
        return Img(), None, None, None, "grid"

    monkeypatch.setattr(sm, "build_mask", fake_build_mask)
    for season in ("Maha", "Yala"):
        calls.clear()
        sm.detection_count(FakeEE, cfg, season)
        assert calls == sm.static_source_seasons(cfg, season)
        assert max(sm.season_order(s, y) for s, y in calls) < first_test_order(cfg)


def _freq():
    rows = []
    for code in ("A", "B"):
        for c, ha in ((0, 500.0), (1, 300.0), (2, 200.0), (3, 100.0)):
            rows.append({"adm2_pcode": code, "season": "Maha", "count": c, "ha": ha})
    return pd.DataFrame(rows)


def _dcs():
    rows = []
    for code in ("A", "B"):
        for y in range(2000, 2026):
            rows.append({"adm2_pcode": code, "season": "Maha", "season_year": y,
                         "sown_ha": 300.0, "yield_kg_ha": 4000.0})
    return pd.DataFrame(rows)


def test_choose_k_ignores_yields_and_later_years(cfg):
    base = sm.choose_k(_freq(), _dcs(), cfg)
    d = _dcs()
    d["yield_kg_ha"] = -1.0                                   # yields must not matter
    d.loc[d.season_year > 2008, "sown_ha"] = 1e9              # post-source years must not matter
    assert sm.choose_k(_freq(), d, cfg)["Maha"]["k"] == base["Maha"]["k"]


def test_choose_k_rule(cfg):
    # pooled target = 600 ha; area >= k: k=1 1200, k=2 600, k=3 200 -> k=2
    r = sm.choose_k(_freq(), _dcs(), cfg)
    assert r["Maha"]["k"] == 2
