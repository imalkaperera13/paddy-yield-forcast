"""Step 4 leakage tests on the REAL masked master table and its inputs.

They skip when the data are not present (e.g. on Kaggle without the raw series).
1. Table shape: one row per district x season x lead; no forbidden feature columns;
   masked and unmasked tables have identical rows; forecast_date = harvest_ref - lead.
2. No feature uses data after the cutoff: rebuilding sampled rows from series truncated at
   the forecast date gives exactly the same features as the full series.
3. No future-season yield: changing every label that is not yet published at the forecast
   date (which includes the target season and all later seasons) changes no feature.
4. Anomalies use strictly earlier years of the same district and season only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import REPO_ROOT, load_config
from src.data.features import Calendar, build_season_features

PROC = REPO_ROOT / "data" / "processed"
MASTER = PROC / "master_masked.parquet"
UNMASKED = PROC / "master_unmasked_7d.parquet"
LABELS = REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv"
KEY = ["district", "season", "season_year", "lead_weeks"]

pytestmark = pytest.mark.skipif(not MASTER.exists(), reason="master_masked.parquet not built")

CFG = load_config()
CAL = Calendar.from_cfg(CFG)


@pytest.fixture(scope="module")
def master():
    return pd.read_parquet(MASTER)


@pytest.fixture(scope="module")
def inputs():
    from src.data.build_masked_features import load_inputs
    try:
        return {s: load_inputs(s) for s in ("Maha", "Yala")}
    except SystemExit:
        pytest.skip("masked series not present")


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(LABELS)


def sample_rows(master, n=12, seed=0):
    """Rows spread over leads, seasons and districts, including late years."""
    k = n // 6 + 1
    idx = [g.sample(min(len(g), k), random_state=seed).index
           for _, g in master.groupby(["lead_weeks", "season"])]
    return master.loc[[i for ix in idx for i in ix]].reset_index(drop=True)


def feature_cols(df):
    skip = {"district", "adm2_pcode", "season", "season_label", "season_year", "harvest_year",
            "season_order", "yield_kg_ha", "mahaweli_merged", "sanity_flag", "forecast_date",
            "lead_weeks", "shift_weeks", "mask", "mahaweli_affected",
            "diag_post_merge_affected"}
    return [c for c in df.columns if c not in skip and not c.endswith("_anom")]


def one_row(r, labels, inp):
    t = pd.DataFrame([{"district": r.district, "adm2_pcode": r.adm2_pcode, "season": r.season,
                       "season_year": r.season_year}])
    return build_season_features(t, labels, inp["rain"], inp["modis"], CAL, int(r.lead_weeks),
                                 0, era5=inp["era5"],
                                 heat_thr=CFG["features"]["heat_day_tmax_c"]).iloc[0]


def assert_same(a, b, cols, what):
    for c in cols:
        x, y = a.get(c, np.nan), b.get(c, np.nan)
        if pd.isna(x) and pd.isna(y):
            continue
        assert x == pytest.approx(y, rel=1e-9, abs=1e-9), f"{what}: {c} {x} != {y}"


# ---------------------------------------------------------------- 1. table shape
def test_one_row_per_district_season_lead(master):
    assert not master.duplicated(KEY).any()
    assert set(master.lead_weeks) == {12, 8, 4}


def test_no_forbidden_feature_columns(master):
    allowed_meta = {"yield_kg_ha", "mahaweli_merged", "season_year", "harvest_year",
                    "season_order"}
    assert not (set(CFG["forbidden_features"]) - allowed_meta) & set(master.columns)


def test_masked_and_unmasked_rows_identical(master):
    if not UNMASKED.exists():
        pytest.skip("unmasked table not built")
    u = pd.read_parquet(UNMASKED)
    assert (master[KEY].values == u[KEY].values).all()


def test_forecast_date_is_harvest_ref_minus_lead(master):
    for r in master.itertuples():
        assert pd.Timestamp(r.forecast_date) == CAL.forecast_date(r.season, r.season_year,
                                                                  r.lead_weeks)


# ---------------------------------------------------------------- 2. nothing after cutoff
def test_truncating_series_at_cutoff_changes_nothing(master, inputs, labels):
    for r in sample_rows(master).itertuples():
        inp = inputs[r.season]
        fdate = pd.Timestamp(r.forecast_date)
        cut = {
            "rain": inp["rain"][inp["rain"].pentad_end <= fdate],
            "modis": inp["modis"][inp["modis"].end <= fdate],
            "era5": inp["era5"][inp["era5"].date < fdate],
        }
        full = one_row(r, labels, inp)
        trunc = one_row(r, labels, cut)
        cols = feature_cols(master)
        assert_same(full, trunc, cols, f"{r.district} {r.season} {r.season_year} L{r.lead_weeks}")
        stored = master[(master.district == r.district) & (master.season == r.season)
                        & (master.season_year == r.season_year)
                        & (master.lead_weeks == r.lead_weeks)].iloc[0]
        assert_same(full, stored, cols, "rebuilt vs stored")


# ---------------------------------------------------------------- 3. no future yields
def test_unpublished_labels_do_not_change_features(master, inputs, labels):
    lag = pd.DateOffset(months=CAL.lag_months)
    rng = np.random.default_rng(0)
    for r in sample_rows(master, seed=1).itertuples():
        fdate = pd.Timestamp(r.forecast_date)
        unpublished = np.array([CAL.harvest_ref(s, y) + lag > fdate
                                for s, y in zip(labels.season, labels.season_year, strict=True)])
        target_order = labels[(labels.district == r.district) & (labels.season == r.season)
                              & (labels.season_year == r.season_year)].season_order.iloc[0]
        assert unpublished[labels.season_order >= target_order].all(), \
            "the target season and later seasons must count as unpublished"
        noisy = labels.copy()
        for c in ("yield_kg_ha", "sown_ha", "harvested_ha", "production_t"):
            noisy.loc[unpublished, c] = rng.uniform(1, 1e5, unpublished.sum())
        a = one_row(r, labels, inputs[r.season])
        b = one_row(r, noisy, inputs[r.season])
        assert_same(a, b, feature_cols(master), f"labels {r.district} {r.season} {r.season_year}")


# ---------------------------------------------------------------- 4. anomalies past-only
def test_modis_anomalies_use_earlier_years_only(master):
    for (dist, season, lead), g in master.groupby(["district", "season", "lead_weeks"]):
        g = g.sort_values("season_year")
        for col in ("ndvi_b1", "evi_b1"):
            vals = g[col].to_numpy()
            anom = g[f"{col}_anom"].to_numpy()
            for i in range(len(g)):
                past = vals[:i][~np.isnan(vals[:i])]
                if len(past) < CFG["features"]["anomaly_min_history_years"] or np.isnan(vals[i]):
                    assert np.isnan(anom[i]), f"{dist} {season} L{lead} {col} row {i}"
                else:
                    assert anom[i] == pytest.approx(vals[i] - past.mean(), abs=1e-9)
