"""Leakage tests for feature building. All inputs here are SYNTHETIC."""

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.features import (
    Calendar,
    add_past_anomalies,
    build_season_features,
    chirps_daily,
    modis_clean,
)

CFG = load_config()
CAL = Calendar.from_cfg(CFG)


def synthetic_chirps(pcode="LK00", start="1999-01-01", end="2003-12-31", value=10.0):
    """SYNTHETIC CHIRPS-like pentads (6 per month, last one runs to month end)."""
    rows = []
    for m in pd.date_range(start, end, freq="MS"):
        starts = [m + pd.Timedelta(days=5 * k) for k in range(6)]
        ends = starts[1:] + [m + pd.offsets.MonthBegin(1)]
        rows += [{"adm2_pcode": pcode, "start": s.date(), "end": e.date(), "precipitation": value}
                 for s, e in zip(starts, ends, strict=True)]
    return pd.DataFrame(rows)


def synthetic_modis(pcode="LK00", start="1999-01-01", end="2003-12-31", ndvi=0.5):
    d = pd.date_range(start, end, freq="16D")
    return pd.DataFrame({"adm2_pcode": pcode, "date": d.strftime("%Y-%m-%d"), "NDVI_mean": ndvi,
                         "EVI_mean": ndvi / 2, "NDVI_count": 100})


def labels_synthetic():
    rows = []
    for y in range(1990, 2004):
        for s in ("Maha", "Yala"):
            rows.append({"district": "D", "season": s, "season_year": y,
                         "yield_kg_ha": 3000 + y + (500 if s == "Yala" else 0),
                         "sown_ha": 1000.0, "harvested_ha": 950.0})
    return pd.DataFrame(rows)


TARGET = pd.DataFrame([{"district": "D", "adm2_pcode": "LK00", "season": "Maha",
                        "season_year": 2001}])


def _features(chirps, modis, lead=0, shift=0, labels=None):
    return build_season_features(TARGET, labels if labels is not None else labels_synthetic(),
                                 chirps_daily(chirps), modis_clean(modis, None), CAL, lead, shift
                                 ).iloc[0]


def test_SYNTHETIC_future_climate_does_not_change_features():
    base_c, base_m = synthetic_chirps(), synthetic_modis()
    f0 = _features(base_c, base_m, lead=6)
    fdate = CAL.forecast_date("Maha", 2001, 6)
    lat_c = pd.Timedelta(days=CAL.latency_days["chirps"])
    lat_m = pd.Timedelta(days=CAL.latency_days["modis"])
    c2, m2 = base_c.copy(), base_m.copy()
    # perturb ONLY data that would not yet be available at the forecast date
    c2.loc[pd.to_datetime(c2.end) + lat_c > fdate, "precipitation"] = 9999
    m2.loc[pd.to_datetime(m2.date) + pd.Timedelta(days=16) + lat_m > fdate, "NDVI_mean"] = 0.99
    f1 = _features(c2, m2, lead=6)
    cols = [c for c in f0.index if c.startswith(("rain_", "ndvi_", "evi_"))]
    pd.testing.assert_series_equal(f0[cols].astype(float), f1[cols].astype(float))


def test_SYNTHETIC_rain_block_sums_are_exact():
    f = _features(synthetic_chirps(value=10.0), synthetic_modis(), lead=0)
    # every pentad = 10 mm -> 2 mm/day for 5-day pentads; a 28-day block sums ~ 56-62 mm
    assert 50 < f["rain_b1"] < 65
    assert np.isclose(f["rain_season"], sum(f[f"rain_b{k}"] for k in range(1, 6)))


def test_label_lag_rule_applied_exactly():
    lab = labels_synthetic()
    lag = pd.DateOffset(months=CFG["data"]["label_availability_lag_months"])
    f0 = _features(synthetic_chirps(), synthetic_modis(), lead=0, labels=lab)
    # same-season previous year (Maha 2000/01) is available
    assert f0["y_prev_same"] == 3000 + 2000
    # the opposite season just before (Yala 2001) is available iff harvest_ref + lag <= fdate
    yala01 = 3000 + 2001 + 500
    for lead in (0, 2, 6):
        f = _features(synthetic_chirps(), synthetic_modis(), lead=lead, labels=lab)
        avail = CAL.harvest_ref("Yala", 2001) + lag <= CAL.forecast_date("Maha", 2001, lead)
        assert (f["y_last_avail"] == yala01) == avail, lead
    # the target season itself and any later season never contribute
    lab2 = lab.copy()
    lab2.loc[(lab2.season_year > 2001) | ((lab2.season_year == 2001) & (lab2.season == "Maha")),
             "yield_kg_ha"] = 1e9
    f2 = _features(synthetic_chirps(), synthetic_modis(), lead=0, labels=lab2)
    for c in ("y_prev_same", "y_mean_same", "y_last5_same", "y_trend_same", "y_last_avail"):
        assert f2[c] < 1e8, c


def test_anomalies_use_only_previous_years():
    df = pd.DataFrame({"district": "D", "season": "Maha", "season_year": range(2000, 2010),
                       "x": [1.0] * 10})
    a = add_past_anomalies(df, ["x"], min_history=3)
    df2 = df.copy()
    df2.loc[df2.season_year == 2009, "x"] = 1000.0     # change only the LAST year
    b = add_past_anomalies(df2, ["x"], min_history=3)
    early = a.season_year < 2009
    pd.testing.assert_series_equal(a.loc[early, "x_anom"], b.loc[early, "x_anom"])
    assert a.loc[a.season_year < 2003, "x_anom"].isna().all()   # < 3 past years -> NaN


def test_forbidden_columns_listed():
    forb = set(CFG["forbidden_features"])
    assert {"yield_kg_ha", "production_t", "sown_ha", "harvested_ha",
            "mahaweli_merged"} <= forb


def test_chirps_column_named_mean_is_accepted():
    c = synthetic_chirps().rename(columns={"precipitation": "mean"})
    assert chirps_daily(c).rate.sum() > 0
